from ..builder import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decode_head import BaseDecodeHead
from ..utils.crss_layer import *
import math
from timm.models.layers import  trunc_normal_
from mmseg.utils.logger import get_root_logger
from mmseg.ops import resize
from packaging import version


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sizes = sizes

    def forward(self, x):
        y = x.permute((0, 3, 1, 2))
        return y


def load_ckpt(ckpt_path):
    torch_version = torch.__version__
    if version.parse(torch_version) >= version.parse("2.0.0"):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"Loaded checkpoint with weights_only=True (PyTorch {torch_version})")
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"Loaded checkpoint without weights_only (PyTorch {torch_version})")
    return ckpt

@HEADS.register_module()
class MixHead(BaseDecodeHead):
    """Mix Head"""
    def __init__(self,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 num_classes = 2,
                 in_index=[0, 1, 2, 3],
                 in_channels=[96, 192, 384, 768],
                 mlp_ratios=[2, 2, 2, 2],
                 num_heads=[2, 4, 8, 16],
                 channels=256,
                 drop_path_rate=0.08,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 ckpt_path="weights/mix_ade_pretrained.pth",
                 align_corners=False,
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         input_transform='multiple_select',
                         in_channels=in_channels,
                         channels=channels,
                         in_index=in_index,
                        **kwargs)
        assert len(in_index) == len(in_channels), f"numbers of in_index:{in_index} not equal numbers of in_channels:{in_channels}"
        self.num_features = len(in_channels)
        self.ffts = nn.ModuleList()
        self.ffts_2 = nn.ModuleList()
        self.lvl_cross = nn.ModuleList()
        self.level_embeds = nn.ModuleList()
        self.convert = nn.ModuleList()
        # self.ssd = nn.ModuleList()
        self.in_index = in_index[::-1]
        self.in_channels = in_channels[::-1]
        self.num_heads = num_heads[::-1]
        self.drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_features)][::-1]
        self.mlp_ratios = mlp_ratios[::-1]
        ## 高频特征增强
        ## 特征位置编码与可学习变量
        for i_layer in range(self.num_features):
            self.convert.append(nn.Conv2d(self.in_channels[i_layer], channels, 1))
            self.ffts.append(HFFTAttention(channels, norm_cfg))
            self.level_embeds.append(nn.Embedding(1, channels))
            self.ffts_2.append(HFFTAttention(channels, norm_cfg))
        self.in_channels = [channels] * self.num_features
        # self.ssd.append(SSD(channels, mlp_ratio=self.mlp_ratios[i_layer], drop_path=self.drop_path_rates[i_layer]))
        # 特征交叉查询补足
        # 上采样 拼接
        for i, dim in enumerate(self.in_channels[:-1]):
            lvl_block = SplitCross(self.in_channels[i + 1:], self.in_channels[i], self.num_heads[i], self.mlp_ratios[i], self.drop_path_rates[i])
            up = UpSample()
            cat = nn.Conv2d(self.in_channels[i] + self.in_channels[i + 1], self.in_channels[i + 1], 3, 1, 1)
            self.lvl_cross.append(lvl_block)
            self.lvl_cross.append(up)
            self.lvl_cross.append(cat)
            ## 全局特征联系 ssd
        self.lvl_cross.append(SelfBlock(self.in_channels[-1], self.num_heads[-1], self.mlp_ratios[-1], self.drop_path_rates[-1]))
        # 特征融合输出
        self.channel_donw = nn.Conv2d(self.num_features * channels, channels, 1, 1, 0)
        self.ssd = SSD(channels, mlp_ratio=2, drop_path=0.05)
        self.fuse = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm = build_norm_layer(norm_cfg, channels)[1]
        self.act = nn.GELU()
        self.apply(self._init_weights)
        self.load_pretrained_ckpt(ckpt_path)
    
    def _init_weights(self, m):
        if getattr(m, "_no_reinit", False) or (hasattr(m, 'bias') and getattr(m.bias, '_no_reinit', False)):
            return
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def load_pretrained_ckpt(self, ckpt_path=None):
        if ckpt_path is None:
            return
        ckpt = load_ckpt(ckpt_path)
        model_dict = self.state_dict()
        pretrained_dict = ckpt['state_dict']
        # 过滤操作
        new_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith("decode_head."):
                k = k.replace("decode_head.", "")
                if k in model_dict.keys() and model_dict[k].shape == v.shape:
                    new_dict[k] = v
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
        self.load_state_dict(model_dict)
    

    def _forward_feature(self, inputs):
        # inputs = self._transform_inputs(inputs)
        assert self.num_features == len(inputs), f"self.num_features :{self.num_features} not equal numbers of inputs:{len(inputs)}"
        inputs = inputs[::-1]
        for i_layer in range(self.num_features):
            x = inputs[i_layer]
            x = self.convert[i_layer](x)
            x = self.ffts[i_layer](x)
            lvl_embed = self.level_embeds[i_layer].weight[0]
            inputs[i_layer] = x + lvl_embed.reshape((1, -1, 1, 1))

        for i_layer in range(self.num_features - 1):
            x = inputs[i_layer]
            x_pre = inputs[i_layer + 1]
            y = inputs[i_layer + 1:]
            x = self.lvl_cross[i_layer * 3](x, y)
            inputs[i_layer] = self.ffts_2[i_layer](inputs[i_layer])
            # inputs[i_layer] = self.ssd[i_layer](inputs[i_layer])
            x_up = self.lvl_cross[i_layer * 3 + 1](x)
            x_pre = torch.cat([x_pre, x_up], dim=1)
            x_pre = self.lvl_cross[i_layer * 3 + 2](x_pre)
            inputs[i_layer + 1] = x_pre
        inputs[-1] = self.lvl_cross[-1](inputs[-1])
        inputs[-1] = self.ffts_2[-1](inputs[-1])
        # inputs[-1] = self.ssd[-1](inputs[-1])
        for i in range(self.num_features - 1):
            inputs[i] = resize(
                inputs[i],
                size=inputs[-1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        inputs = torch.cat(inputs, dim=1)
        rs = self.channel_donw(inputs)
        rs = self.ssd(rs)
        rs = self.act(self.norm(self.fuse(rs)))
        return rs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

# if __name__ == "__main__":
#     device = 'cuda'
#     B = 8
#     H, W = 128, 128
#     in_channels = [96,192,384,768]
#     inputs = [torch.randn(B,c,H//(2**i), W//(2**i), device=device) for i,c in enumerate(in_channels)]
#     model = MixHead(in_channels=in_channels).to(device)

#     torch.cuda.reset_peak_memory_stats()
#     with torch.no_grad():
#         output = model(inputs)

#     print("Output shape:", output.shape)
#     print("Max GPU memory allocated (GB):", torch.cuda.max_memory_allocated()/1024**3)
#     print("Current GPU memory allocated (GB):", torch.cuda.memory_allocated()/1024**3)
