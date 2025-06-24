import torch
import torch.nn as nn
import math
import copy

from timm.models.layers import trunc_normal_
from mmcv.runner import BaseModule
from ..builder import BACKBONES
from ..utils.vssm_layer import PatchEmbed2D, PatchMerging2D, VSSLayer
from mmseg.utils.logger import get_root_logger


@BACKBONES.register_module()
class CrsCmt(BaseModule):
    """
    MMSegmentation兼容的Mamba主干网络
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768],
                 d_state=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 out_indices=(0, 1, 2, 3),
                 pretrained=None,
                 ape=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.out_indices = out_indices
        self.ape = ape
        # 图像到patch的转换
        
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]



    def _init_weights(self, m):
        pass

    def init_weights(self):
        pass

    def forward(self, x):
        pass
