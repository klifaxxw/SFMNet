import time
import math
import copy
from builtins import int
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.runner import BaseModule


class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

# 多层感受野
class MRFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(MRFBlock, self).__init__()



class BFEBlock(nn.Module):
    def __init__(self, in_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(BFEBlock, self).__init__()


# 边缘特征增强
class BFAttention(nn.Module):
    def __init__(self, in_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(BFAttention, self).__init__()


class MFVssBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(MFVssBlock, self).__init__()





