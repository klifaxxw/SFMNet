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
from ..utils.rs_layer import Decoder_Block

# Code adapted from RS_mamba: https://github.com/walking-shadow/Official_Remote_Sensing_Mamba.git
# modified output from B 1 H W to B C H W

@HEADS.register_module()
class RSMHead(BaseDecodeHead):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 in_index=[0, 1, 2, 3],
                 channels=128,
                 num_classes=150,
                 norm_layer="sn",
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         input_transform='multiple_select',
                         in_channels=in_channels,
                         channels=channels,
                         in_index=in_index,)
        _NORMLAYERS = dict(ln=nn.LayerNorm, bn=nn.BatchNorm2d, sn=nn.SyncBatchNorm)
        norm_layer = _NORMLAYERS[norm_layer.lower()]
        self.decoder3 = Decoder_Block(in_channel=in_channels[3], out_channel=in_channels[2], norm_layer=norm_layer)
        self.decoder2 = Decoder_Block(in_channel=in_channels[2], out_channel=in_channels[1], norm_layer=norm_layer)
        self.decoder1 = Decoder_Block(in_channel=in_channels[1], out_channel=in_channels[0], norm_layer=norm_layer)
        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0] // 2, kernel_size=3, stride=1, padding=1),
            norm_layer(in_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels[0] // 2, in_channels[0] // 2, kernel_size=3, stride=1, padding=1),
            norm_layer(in_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.conv_out_seg = nn.Conv2d(in_channels[0] // 2, self.num_classes, kernel_size=7, stride=1, padding=3)

    def _forward_feature(self, inputs):
        x1, x2, x3, x4 = self._transform_inputs(inputs)
        d3 = self.decoder3(x4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)
        out = self.upsample_x4(d1)
        out = self.conv_out_seg(out)
        return out

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        return output
