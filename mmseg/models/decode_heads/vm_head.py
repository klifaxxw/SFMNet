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
from ..utils.vm_layer import *
from packaging import version


# Code adapted from RS_mamba: https://github.com/walking-shadow/Official_Remote_Sensing_Mamba.git
# modified output from B 1 H W to B C H W

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
class VMUHead(BaseDecodeHead):
    def __init__(self,
                 in_channels=[768, 384, 192, 96],
                 depths=[2, 9, 2, 2],
                 in_index=[0, 1, 2, 3],
                 d_state=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 pretrained_ckpt="weights/vm_unet_tiny_e292.pth",
                 channels=64,
                 num_classes=150,
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         input_transform='multiple_select',
                         in_channels=in_channels,
                         channels=channels,
                         in_index=in_index)
        
        self.num_layers = len(in_channels)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=in_channels[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(in_channels[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=in_channels[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(in_channels[-1] // 4, num_classes, 1)
        self.apply(self._init_weights)
        self.load_pretrained_ckpt(pretrained_ckpt)
        
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained_ckpt(self, ckpt_path=None):
        if ckpt_path is None:
            return
        ckpt = load_ckpt(ckpt_path)
        model_dict = self.state_dict()
        pretrained_odict = ckpt['model']
        # 过滤操作
        pretrained_dict = {}
        for k, v in pretrained_odict.items():
            if 'layers.0' in k: 
                new_k = k.replace('layers.0', 'layers_up.3')
                pretrained_dict[new_k] = v
            elif 'layers.1' in k: 
                new_k = k.replace('layers.1', 'layers_up.2')
                pretrained_dict[new_k] = v
            elif 'layers.2' in k: 
                new_k = k.replace('layers.2', 'layers_up.1')
                pretrained_dict[new_k] = v
            elif 'layers.3' in k: 
                new_k = k.replace('layers.3', 'layers_up.0')
                pretrained_dict[new_k] = v
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
        self.load_state_dict(model_dict)
        # 找到没有加载的键(keys)
        not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
        print('Not loaded keys:', not_loaded_keys)
        print("decoder loaded finished!")
    
    def _forward_feature(self, inputs):
        x, y = inputs[-1], inputs[:-1]
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + y[-inx])
        x = self.final_up(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        return output
