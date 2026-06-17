import torch
import torch.nn as nn

from mmcv.cnn import build_norm_layer
from ..builder import HEADS
from .decode_head import BaseDecodeHead


def build_norm(norm_cfg, num_features, postfix=''):
    return build_norm_layer(norm_cfg, num_features, postfix=postfix)[1]


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, norm_cfg, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan,
            kernel_size=ks, stride=stride, padding=padding,
            bias=False
        )
        self.bn = build_norm(norm_cfg, out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


@HEADS.register_module()
class ABCOutputHead(BaseDecodeHead):
    """
    1:1 复刻原 Output:
        ConvBNReLU(in, mid, 3x3) ->
        Conv2d(mid, num_classes * r^2, 1x1) ->
        PixelShuffle(r)
    """
    def __init__(
        self,
        mid_channels=256,
        up_factor=8,
        norm_cfg=dict(type='BN', requires_grad=True),
        **kwargs
    ):
        super().__init__(input_transform=None, **kwargs)

        self.mid_channels = mid_channels
        self.up_factor = up_factor
        self.norm_cfg = norm_cfg

        self.conv = ConvBNReLU(self.in_channels, mid_channels, norm_cfg=norm_cfg, ks=3, stride=1, padding=1)

        # 原 Output.conv_out
        out_chan = self.num_classes * up_factor * up_factor
        self.conv_out = nn.Conv2d(mid_channels, out_chan, kernel_size=1, bias=True)

        # 原 Output.up
        self.up = nn.PixelShuffle(up_factor)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x
