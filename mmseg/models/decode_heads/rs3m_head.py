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
from ..utils.rs3_layer import *

# Code adapted from RS_mamba: https://github.com/walking-shadow/Official_Remote_Sensing_Mamba.git
# modified output from B 1 H W to B C H W

@HEADS.register_module()
class RS3MHead(BaseDecodeHead):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 in_index=[0, 1, 2, 3],
                 decode_channels=64,
                 dropout=0.1,
                 channels=64,
                 window_size=8,
                 num_classes=150,
                 norm_layer="sb",
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         input_transform='multiple_select',
                         in_channels=in_channels,
                         channels=channels,
                         in_index=in_index)
        _NORMLAYERS = dict(ln=nn.LayerNorm, bn=nn.BatchNorm2d, sb=nn.SyncBatchNorm)
        norm_layer = _NORMLAYERS[norm_layer.lower()]
        self.decoder = Decoder(in_channels, decode_channels, dropout, window_size, num_classes, norm_layer=norm_layer)       

    def _forward_feature(self, inputs):
        x1, x2, x3, x4 = self._transform_inputs(inputs)
        b, c, h1, w1 = x1.shape
        h, w = h1 * 4, w1 * 4 
        out = self.decoder(x1, x2, x3, x4, h, w)
        return out

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        return output
