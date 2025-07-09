from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import math
import copy

from timm.models.layers import  trunc_normal_
from mmcv.runner import BaseModule
from ..builder import BACKBONES
from ..utils.vssm_layer import PatchEmbed2D, PatchMerging2D, VSSLayer
from mmseg.utils.logger import get_root_logger

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sizes = sizes

    def forward(self, x):
        y = x.permute((0, 3, 1, 2))
        return y


@BACKBONES.register_module()
class MambaUE(BaseModule):
    """
    MMSegmentation兼容的Mamba主干网络
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768],
                 d_state=16,
                 ssm_ratio=1,
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

        # 构建编码器层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=int(dims[0] * 2 ** i_layer),
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            setattr(self, f"reshape{i_layer + 1}", Reshape())
        
        self.reshape5 = Reshape()  
        # 输出层的归一化
        self.norm = norm_layer(self.num_features)

        # 初始化权重
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if hasattr(m, 'bias') and getattr(m.bias, '_no_reinit', False):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def init_weights(self):
        logger = get_root_logger()
        """自定义初始化权重"""
        logger.info("init_weights is called.")
        if self.init_cfg is not None:
            checkpoint = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            # 检查是否包含 'model' 键
            if 'model' in checkpoint:
                logger.info("Checkpoint contains 'model'. Extracting state_dict...")
                state_dict = checkpoint['model']
            else:
                logger.info("Checkpoint does not contain 'model'. Loading directly...")
                state_dict = checkpoint
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.info("No init_cfg provided. Using default initialization.")

    def forward(self, x):
        """前向传播"""
        x = self.patch_embed(x)  # B C H W -> B H W C
        ### position embeding
        if self.ape:
            pass
        outs = []
        for i, layer in enumerate(self.layers):
            reshape = getattr(self, f"reshape{i + 1}")
            # y = x.permute((0, 3, 1, 2))
            x = reshape(x)
            outs.append(x)
            x = x.permute((0, 2, 3, 1))
            x = layer(x)
        x = self.norm(x)
        y = self.reshape5(x)
        outs.append(y)
        return tuple(outs)
