"""
BiFormer impl (MMSeg backbone version, mmseg0.24).

- Converted from ImageNet classification code to MMSeg backbone.
- Removed classification head / global pooling.
- Forward returns multi-stage features (tuple of NCHW tensors).
- Registered into MMSeg BACKBONES registry.
- Supports init_cfg Pretrained loading with strict=False (filters head/pre_logits).

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import LayerNorm2d

# ===== MMSeg / MMCV =====
from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer
from mmseg.models.builder import BACKBONES

import re
from packaging import version

# ===== Your ops / common =====
from ..utils import BiLevelRoutingAttention
from ..utils import Attention, AttentionLePE, DWConv

def load_ckpt(ckpt_path):
    torch_version = torch.__version__
    if version.parse(torch_version) >= version.parse("2.0.0"):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"Loaded checkpoint with weights_only=True (PyTorch {torch_version})")
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"Loaded checkpoint without weights_only (PyTorch {torch_version})")
    return ckpt


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none' or name is None:
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            pad = before_attn_dwconv // 2
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=pad, groups=dim)
        else:
            self.pos_embed = lambda x: 0

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing

        if topk > 0:
            self.attn = BiLevelRoutingAttention(
                dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                topk=topk, param_attention=param_attention, param_routing=param_routing,
                diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                auto_pad=auto_pad
            )
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(
                Rearrange('n h w c -> n c h w'),     # compatibility
                nn.Conv2d(dim, dim, 1),             # pseudo qkv linear
                nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                nn.Conv2d(dim, dim, 1),             # pseudo out linear
                Rearrange('n c h w -> n h w c')
            )
        else:
            raise ValueError(f'Unsupported topk={topk}')

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False

        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))   # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))                # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))                 # (N, H, W, C)
        else:  # post-norm
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))   # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))                # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))                 # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


@BACKBONES.register_module()
class BiFormer(BaseModule):
    def __init__(self,
                 depth=(3, 4, 8, 3),
                 in_chans=3,
                 embed_dim=(64, 128, 320, 512),
                 head_dim=64,
                 qk_scale=None,
                 drop_path_rate=0.,
                 drop_rate=0.,
                 use_checkpoint_stages=(),
                 # -------- BiFormer routing params --------
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=(2, 2, -1, -1),
                 topks=(8, 8, -1, -1),
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=(None, None, None, None),
                 param_routing=False,
                 diff_routing=False,
                 soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=(0,),
                 before_attn_dwconv=3,
                 auto_pad=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 # -------- extra controls --------
                 kv_downsample_kernels=(4, 2, 1, 1),
                 kv_downsample_ratios=(4, 2, 1, 1),
                 mlp_ratios=(4, 4, 4, 4),
                 param_attention='qkvo',
                 mlp_dwconv=False,
                 # -------- mmseg standard --------
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dim = list(embed_dim)
        self.out_indices = tuple(out_indices)

        depth = list(depth)
        kv_per_wins = list(kv_per_wins)
        topks = list(topks)
        qk_dims = list(qk_dims)
        kv_downsample_kernels = list(kv_downsample_kernels)
        kv_downsample_ratios = list(kv_downsample_ratios)
        mlp_ratios = list(mlp_ratios)
        use_checkpoint_stages = list(use_checkpoint_stages)
        pe_stages = list(pe_stages)

        # ------------- downsample layers (patch embeddings) -------------
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, self.embed_dim[0] // 2, kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_cfg, self.embed_dim[0] // 2)[1],
            nn.GELU(),
            nn.Conv2d(self.embed_dim[0] // 2, self.embed_dim[0], kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_cfg, self.embed_dim[0])[1],
        )
        if (pe is not None) and (0 in pe_stages):
            stem.append(get_pe_layer(emb_dim=self.embed_dim[0], name=pe))
        # if use_checkpoint_stages:
        #     stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(self.embed_dim[i], self.embed_dim[i + 1], kernel_size=3, stride=2, padding=1),
                build_norm_layer(norm_cfg, self.embed_dim[i + 1])[1]
            )
            if (pe is not None) and (i + 1 in pe_stages):
                downsample_layer.append(get_pe_layer(emb_dim=self.embed_dim[i + 1], name=pe))
            # if use_checkpoint_stages:
            #     downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)

        # ------------- stages -------------
        self.stages = nn.ModuleList()

        # fix qk_dims=None -> use embed_dim
        qk_dims = [qd if qd is not None else self.embed_dim[i] for i, qd in enumerate(qk_dims)]

        # number of heads per stage
        nheads = [qd // head_dim for qd in qk_dims]

        # stochastic depth
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(
                    dim=self.embed_dim[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    topk=topks[i],
                    num_heads=nheads[i],
                    n_win=n_win,
                    qk_dim=qk_dims[i],
                    qk_scale=qk_scale,
                    kv_per_win=kv_per_wins[i],
                    kv_downsample_ratio=kv_downsample_ratios[i],
                    kv_downsample_kernel=kv_downsample_kernels[i],
                    kv_downsample_mode=kv_downsample_mode,
                    param_attention=param_attention,
                    param_routing=param_routing,
                    diff_routing=diff_routing,
                    soft_routing=soft_routing,
                    mlp_ratio=mlp_ratios[i],
                    mlp_dwconv=mlp_dwconv,
                    side_dwconv=side_dwconv,
                    before_attn_dwconv=before_attn_dwconv,
                    pre_norm=pre_norm,
                    auto_pad=auto_pad
                ) for j in range(depth[i])]
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))

        # init (only for modules created here)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        """Support mmseg init_cfg pretrained loading with tolerant key matching."""
        super().init_weights()

        if self.init_cfg is None:
            return
        if self.init_cfg.get('type', None) != 'Pretrained':
            return

        ckpt = self.init_cfg.get('checkpoint', None)
        if ckpt is None:
            return

        state = load_ckpt(ckpt)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']

        if isinstance(state, dict):
            state = {k: v for k, v in state.items()
                     if not (k.startswith('head.') or k.startswith('pre_logits.'))}

        # tolerate mismatch (segmentation removes head)
        self.load_state_dict(state, strict=False)

    def forward(self, x):
        """Return multi-level features: tuple of NCHW tensors."""
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(self.extra_norms[i](x))
        return tuple(outs)
