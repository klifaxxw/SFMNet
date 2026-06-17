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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias))

@HEADS.register_module()
class UNetFormerMainHead(BaseDecodeHead):
    def __init__(self, decode_channels=64, dropout=0.1,
                 norm_cfg=dict(type='BN', requires_grad=True), **kwargs):
        super().__init__(input_transform=None, **kwargs)
        nl = norm_layer_factory(norm_cfg)
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(self.in_channels, decode_channels, norm_layer=nl),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, self.num_classes, kernel_size=1, bias=True),
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        return self.segmentation_head(x)
