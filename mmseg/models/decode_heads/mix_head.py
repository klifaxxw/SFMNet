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


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sizes = sizes

    def forward(self, x):
        y = x.permute((0, 3, 1, 2))
        return y

### cross attention
class CrossBlock(nn.Module):
    def __init__(self, in_channels, ebdim, n_head, mlp_ratio, drop_path_rate, kernel_size=7, reduction = 16, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.rope = RoPE(ebdim, n_head)
        self.pre = nn.ModuleList()
        self.cr = nn.ModuleList()
        self.kv_proj = nn.ModuleList()
        self.q_proj = nn.Linear(ebdim, ebdim)
        self.num_head = n_head
        self.num_features = len(in_channels)
        for i, in_channel in enumerate(in_channels):
            ll = nn.Linear(in_channel, ebdim)
            kv = nn.Linear(ebdim, 2 * ebdim)
            cross = CrossAttention(ebdim, n_head, mlp_ratio, drop_path_rate)
            self.pre.append(ll)
            self.cr.append(cross)
            self.kv_proj.append(kv)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=autopad(kernel_size), bias=False)
        self.se = SEBlock(ebdim, reduction)
        self.norm = build_norm_layer(norm_cfg, ebdim)[0]
        self.act = nn.GELU()
        
    def apply_rotateEnc(self, x, sin, cos, b, c, h, w):
        x = x.reshape((b, h, w, self.num_head, -1)).permute((0, 3, 1, 2, 4)).contiguous()
        x = theta_shift(x, sin, cos).permute((0, 2, 3, 1, 4)).contiguous().reshape((b, h * w, c)) # 旋转编码
        return x
        
    def forward(self, x, inputs):
        b, c, h, w = x.shape
        sin, cos = self.rope((h, w))
        x = x.reshape((b, c, -1)).transpose(1, 2) ### b, l, c
        q = self.q_proj(x) # query
        q = self.apply_rotateEnc(q, sin, cos, b, c, h, w)
        rs = []
        for i_layer in range(self.num_features):
            input = inputs[i_layer]
            input = self.pre[i_layer](input)
            _, _ , h1, w1 = input.shape
            sin1, cos1 = self.rope((h1, w1))
            input = input.reshape((b, c, -1)).transpose(1, 2)
            kv = self.kv_proj[i_layer](input)
            k, v = kv.split(c, dim=-1)
            k = self.apply_rotateEnc(k, sin1, cos1, b, c, h1, w1)
            out = self.cr[i_layer](q, k, v)
            out = out.transpose(1, 2).reshape((b, c, h, w))
            rs.append(out)
        ## balance sum
        rs = torch.stack(rs, dim=1) ## bs, num_features, c, h, w
        rs_e = rs.reshape((-1, c, h, w))
        rs_avg = torch.mean(rs_e, dim=1, keepdim=True)
        rs_max = torch.max(rs_e, dim=1, keepdim=True)[0]
        spatial = torch.cat([rs_avg, rs_max], dim=1)
        spatial = self.conv(spatial)
        spatial = spatial.reshape((b, -1, h, w))
        score = F.softmax(spatial, dim=1).reshape((b, -1, 1, h, w))
        rs = rs * score
        rs = rs.sum(dim=1)
        ## 通道简单变换增强
        rs = self.se(rs)
        rs = self.act(self.norm(rs))
        return rs

class SelfBlock(nn.Module):
    def __init__(self,  ebdim, n_head, mlp_ratio, drop_path_rate):
        super().__init__()
        self.rope = RoPE(ebdim, n_head)
        self.qkv_proj = nn.Linear(ebdim, ebdim)
        self.num_head = n_head
        # self attention
        self.attention = CrossAttention(ebdim, n_head, mlp_ratio, drop_path_rate)
        
    def apply_rotateEnc(self, x, sin, cos, b, c, h, w):
        x = x.reshape((b, h, w, self.num_head, -1)).permute((0, 3, 1, 2, 4)).contiguous()
        x = theta_shift(x, sin, cos).permute((0, 2, 3, 1, 4)).contiguous().reshape((b, h * w, c)) # 旋转编码
        return x
        
    def forward(self, x):
        b, c, h, w = x.shape
        sin, cos = self.rope((h, w))
        x = x.reshape((b, c, -1)).transpose(1, 2) ### b, l, c
        qkv = self.qkv_proj(x) # qkv
        q, k, v = qkv.split(c, dim=-1)
        q = self.apply_rotateEnc(q, sin, cos, b, c, h, w)
        k = self.apply_rotateEnc(k, sin, cos, b, c, h, w)
        rs = self.self.attention(q, k, v)
        return rs

class SSD(nn.Module):
    def __init__(self, ebdim, ssm_ratio=1, norm_cfg=dict(type='LN', requires_grad=True)):
        super().__init__()
        self.ssm = VSSM(d_model=ebdim, expansion_ratio=ssm_ratio)
        self.dwconv = nn.Sequential(
                nn.Conv2d(ebdim, ebdim, kernel_size=3, padding=1, groups=ebdim),
                nn.BatchNorm2d(ebdim),
        )
        self.act = nn.GELU()
        self.norm = build_norm_layer(norm_cfg, ebdim)[1]
        self.conv = nn.Conv2d(ebdim, ebdim, kernel_size=3, padding=1)

    
    def forward(self, x):
        b, c, h, w = x.shape
        v_r = x
        v = self.act(self.dwconv(v_r))
        v = self.ssm(v)
        v = v.permute(0, 2, 1)
        v = self.norm(v)
        v = v.permute(0, 2, 1).reshape((b, -1, h, w))
        x = v_r + v
        return x

@HEADS.register_module()
class MixHead(BaseDecodeHead):
    """Mix Head"""
    def __init__(self,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 num_classes = 2,
                 in_index=[0, 1, 2, 3],
                 in_channels=[96, 192, 384, 768],
                 mlp_ratios=[4, 4, 4, 4],
                 num_heads=[2, 4, 8, 16],
                 channels=256,
                 drop_path_rate=0.08,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
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
        self.ssd = nn.ModuleList()

        self.in_index = in_index[::-1]
        self.in_channels = in_channels[::-1]
        self.num_heads = num_heads[::-1]
        self.drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_features)][::-1]
        self.mlp_ratios = mlp_ratios[::-1]
        ## 高频特征增强
        ## 特征位置编码与可学习变量
        for i_layer in self.in_index:
            self.ffts.append(HFFTAttention(self.in_channels[i_layer], norm_cfg))
            self.level_embeds.append(nn.Embedding(1, self.in_channels[i_layer]))
            self.convert.append(nn.Conv2d(self.in_channels[i_layer], channels, 3, 1))
            self.ffts_2.append(HFFTAttention(channels, norm_cfg))
            self.ssd.append(SSD(channels))
        ## 特征交叉查询补足
        ## 上采样 拼接 
        for i, dim in enumerate(self.in_channels[:-1]):
            lvl_block = CrossBlock(self.in_channels[i + 1:], self.in_channels[i], self.num_heads[i], self.mlp_ratios[i], self.drop_path_rates[i])
            up = UpSample(self.in_channels[i], self.in_channels[i + 1])
            cat = nn.Conv2d(2 * self.in_channels[i + 1], self.in_channels[i + 1], 3, 1)
            self.lvl_cross.append(lvl_block)
            self.lvl_cross.append(up)
            self.lvl_cross.append(cat)
            ## 全局特征联系 ssd
        self.lvl_cross.append(SelfBlock(self.in_channels[-1], self.num_heads[-1], self.mlp_ratios[-1], self.drop_path_rates[-1]))
        ## 特征融合输出
        self.fuse = nn.Conv2d(self.num_features * channels, channels, 3, 1)
        self.norm = build_norm_layer(norm_cfg, channels)[0]
        self.act = nn.GELU()
        self.apply(self._init_weights)
    
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
    
    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)
        assert self.num_features == len(inputs), f"self.num_features :{self.num_features} not equal numbers of inputs:{len(inputs)}"
        inputs = inputs[::-1]
        for i_layer in range(self.num_features):
            x = inputs[i_layer]
            x = self.ffts[i_layer](x)
            lvl_embed = self.level_embeds[i_layer].weight[0]
            inputs[i_layer] = x + lvl_embed.reshape((1, -1, 1, 1))

        for i_layer in range(self.num_features - 1):
            x = inputs[i_layer]
            x_pre = inputs[i_layer + 1]
            y = inputs[i_layer + 1:]
            x = self.lvl_cross[i_layer * 3](x, y)
            inputs[i_layer] = self.convert[i_layer](x)
            inputs[i_layer] = self.ffts_2[i_layer](inputs[i_layer])
            inputs[i_layer] = self.ssd[i_layer](inputs[i_layer])
            x_up = self.lvl_cross[i_layer * 3 + 1](x)
            x_pre = torch.cat([x_pre, x_up], dim=1)
            x_pre = self.lvl_cross[i_layer * 3 + 2](x_pre)
            inputs[i_layer + 1] = x_pre
        
        inputs[-1] = self.lvl_cross[-1](inputs[-1])
        inputs[-1] = self.convert[-1](inputs[-1])
        inputs[-1] = self.ffts_2[-1](inputs[-1])
        inputs[-1] = self.ssd[-1](inputs[-1])

        for i in range(self.num_features - 1):
            inputs[i] = resize(
                inputs[i],
                size=inputs[-1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        inputs = torch.cat(inputs, dim=1)
        rs = self.act(self.norm(self.fuse(inputs)))
        ### 继续加一个ssd
        return rs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
