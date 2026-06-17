# sff_decode_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils import MDAF, FMS

# 复用你原来的 Conv/ConvBNReLU/... 工具
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

# WF (weight fusion) copy
class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

@HEADS.register_module()
class SFFHead(BaseDecodeHead):
    """
    A decode head that implements the SFF-style fusion/head.
    Assumes backbone returns 4 stages: res1, res2, res3, res4 (in that order).
    in_channels_list should match the channels of those stages (list of 4 ints).
    """
    def __init__(self,
                 in_channels,        # can be int or list; we expect list of 4 ints
                 channels=96,
                 num_classes=6,
                 dropout_ratio=0.1,
                 window_size=8,
                 **kwargs):
        super(SFFHead, self).__init__(in_channels=sum(in_channels) if len(in_channels) > 1 else in_channels[0],
                                                channels=channels,
                                                num_classes=num_classes,
                                                dropout_ratio=dropout_ratio,
                                                **kwargs)
        c1, c2, c3, c4 = in_channels
        self.conv2 = ConvBN(c2, channels, kernel_size=1)
        self.conv3 = ConvBN(c3, channels, kernel_size=1)
        self.conv4 = ConvBN(c4, channels, kernel_size=1)

        self.MDAF_L = MDAF(channels, num_heads=8, LayerNorm_type='WithBias')
        self.MDAF_H = MDAF(channels, num_heads=8, LayerNorm_type='WithBias')
        self.fuseFeature = FMS(in_ch=3*channels, out_ch=channels, num_heads=8, window_size=window_size)
        self.WF1 = WF(in_channels=channels, decode_channels=channels)
        self.WF2 = WF(in_channels=channels, decode_channels=channels)

        self.down = Conv(in_channels=3*channels, out_channels=channels, kernel_size=1)
        # segmentation_head similar to original
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Dropout2d(p=dropout_ratio, inplace=True),
            Conv(channels, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) >= 4, "SFFDecodeHead requires at least 4 feature stages"
        res1, res2, res3, res4 = inputs[:4]
        res1h, res1w = res1.size()[-2:]
        # project deeper stages to decode channels
        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        # upsample to res1 spatial size
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        middleres = torch.cat([res2, res3, res4], dim=1)  # (B, 3*channels, H1, W1)
        # fuseFeature returns fusefeature_L, fusefeature_H, glb, local  (same interface as your FMS)
        fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(middleres, None)
        glb = self.MDAF_L(fusefeature_L, glb)
        local = self.MDAF_H(fusefeature_H, local)
        res = self.WF1(glb, local)
        middleres_down = self.down(middleres)  # (B, channels, H1, W1)
        res = F.interpolate(res, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res = middleres_down + res
        res = self.WF2(res, res1)
        out = self.segmentation_head(res)
        return out