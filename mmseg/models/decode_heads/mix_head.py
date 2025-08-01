from ..builder import HEADS
import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ..utils.crss_layer import *
import math
from timm.models.layers import  trunc_normal_
from mmseg.utils.logger import get_root_logger

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sizes = sizes

    def forward(self, x):
        y = x.permute((0, 3, 1, 2))
        return y

@HEADS.register_module()
class MixHead(BaseDecodeHead):
    """Mamba UNet Decode Head"""
    def __init__(self,
                 depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768],
                 d_state=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 num_classes = 2,
                 in_index=[0, 1, 2, 3],
                 in_channels=[96, 192, 384, 768],
                 channels=128,
                 dropout_ratio=0.1,
                 norm_cfg=None,
                 align_corners=False,
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         input_transform='multiple_select',
                         in_channels=in_channels,
                         channels=channels,
                         in_index=in_index,
                        **kwargs)
        
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
        


    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
