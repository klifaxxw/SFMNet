import torch.nn as nn
from mmcv.cnn import build_norm_layer
from ..builder import HEADS
from .decode_head import BaseDecodeHead


def build_norm(norm_cfg, num_features, postfix=''):
    return build_norm_layer(norm_cfg, num_features, postfix=postfix)[1]


def norm_layer_factory(norm_cfg):
    def _make(c):
        return build_norm(norm_cfg, c)
    return _make


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                bias=bias
            ),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias))


@HEADS.register_module()
class UNetFormerAuxHead(BaseDecodeHead):
    """Aux head: ConvBNReLU -> Dropout -> 1x1 Conv"""
    def __init__(self,
                 dropout=0.1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 **kwargs):
        super().__init__(input_transform=None, **kwargs)
        nl = norm_layer_factory(norm_cfg)
        self.conv = ConvBNReLU(self.in_channels, self.in_channels, norm_layer=nl)
        self.drop = nn.Dropout(dropout)
        self.conv_out = Conv(self.in_channels, self.num_classes, kernel_size=1, bias=True)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # pick by in_index
        x = self.conv(x)
        x = self.drop(x)
        x = self.conv_out(x)
        return x
