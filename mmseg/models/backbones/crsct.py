# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn.utils.weight_init import normal_init

from mmcv.runner import load_checkpoint, BaseModule
from mmseg.utils import get_root_logger
from mmcv.cnn import build_norm_layer
from ..builder import BACKBONES
from .swin_transformer import Mlp, SwinTransformerBlock, Reshape, window_partition, PatchEmbed, WindowAttention, \
    window_reverse


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute((0, 3, 1, 2))
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LiteMutilAttention(BaseModule):
    def __init__(self, dim, num_heads, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.conv_q = nn.Conv2d(dim, dim, 1, padding=0, groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, 1, padding=0, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim // num_heads)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim // num_heads)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim // num_heads)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim // num_heads)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 15), padding=(0, 7), groups=dim // num_heads)
        self.conv2_2 = nn.Conv2d(dim, dim, (15, 1), padding=(7, 0), groups=dim // num_heads)

        self.conv3_1 = nn.Conv2d(
            dim, dim, (1, 17), padding=(0, 8), groups=dim // num_heads)
        self.conv3_2 = nn.Conv2d(
            dim, dim, (17, 1), padding=(8, 0), groups=dim // num_heads)
        self.conv3 = nn.Conv2d(4 * dim, dim, 1)
        self.activate = nn.GELU()
        self.norm = build_norm_layer(norm_cfg, dim)[1]

    def forward(self, x, mask_attn=None):
        q = self.conv_q(x)
        v = self.conv_v(x)

        qk_0 = self.conv0_1(q)
        qk_0 = self.conv0_2(qk_0)
        x_0 = v * torch.softmax(qk_0, dim=1)

        qk_1 = self.conv1_1(q)
        qk_1 = self.conv1_2(qk_1)
        x_1 = v * torch.softmax(qk_1, dim=1)

        qk_2 = self.conv2_1(q)
        qk_2 = self.conv2_2(qk_2)
        x_2 = v * torch.softmax(qk_2, dim=1)

        qk_3 = self.conv3_1(q)
        qk_3 = self.conv3_2(qk_3)
        x_3 = v * torch.softmax(qk_3, dim=1)
        x_cat = torch.cat([x_0, x_1, x_2, x_3], dim=1)
        x_e = self.conv3(x_cat)
        o = self.activate(self.norm(x_e))
        return o



class AttentionModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (15, 1), padding=(7, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


class SpatialAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        x = x.permute((0, 2, 3, 1))
        return x


class LiteCTransBlock(BaseModule):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = LiteMutilAttention(dim, num_heads=num_heads, norm_cfg=norm_cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_attn):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute((0, 3, 1, 2))
        attn = self.attn(x, mask_attn)  # nW*B, window_size*window_size, C
        x = attn.view(B, C, H * W).permute((0, 2, 1))
        # FFN 前向回馈网络
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossSwinTransformerBlock(BaseModule):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        # if shift_size > 0:
        #     ### 交叉感知模块
        #     self.spatialAttention = SpatialAttention(dim)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # cyclic shift, 窗口移动
        if self.shift_size > 0:
            # pad feature maps to multiples of window size
            # self.spatialAttention.H = H
            # self.spatialAttention.W = W
            # x = self.spatialAttention(x)
            pad_l = pad_t = 0
            double_window_size = self.window_size * 2
            pad_r = (double_window_size - W % double_window_size) % double_window_size
            pad_b = (double_window_size - H % double_window_size) % double_window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape
            Hp = Hp // 2
            Wp = Wp // 2
            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
            x_n = torch.cat([x0, x1, x2, x3], 0)
            shifted_x = torch.roll(x_n, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            # pad feature maps to multiples of window size
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x0, x1, x2, x3 = torch.chunk(shifted_x, chunks=4, dim=0)
            x[:, 0::2, 0::2, :] = x0
            x[:, 1::2, 0::2, :] = x1
            x[:, 0::2, 1::2, :] = x2
            x[:, 1::2, 1::2, :] = x3
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        # FFN 前向回馈网络
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LiteBasicLayer(BaseModule):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) if (i % 2 == 0) else
            LiteCTransBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                norm_cfg=norm_cfg,
            )
            for i in range(depth)
        ])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None)
            else:
                x = blk(x, None)
        if self.downsample is not None:
            x = x
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class CrossBasicLayer(BaseModule):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 4
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            CrossSwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        double_window_size = self.window_size * 2
        Hp = int(np.ceil(H / double_window_size)) * self.window_size
        Wp = int(np.ceil(W / double_window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


@BACKBONES.register_module()
class CCrossTransformer(BaseModule):
    def __init__(self,
                 pretrain_img_size=512,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 num_lite_stage=2,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_lite_stage = num_lite_stage
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # if i_layer < self.num_lite_stage:
            #     layer = LiteBasicLayer(
            #         dim=int(embed_dim * 2 ** i_layer),
            #         depth=depths[i_layer],
            #         num_heads=num_heads[i_layer],
            #         window_size=window_size,
            #         mlp_ratio=mlp_ratio,
            #         qkv_bias=qkv_bias,
            #         qk_scale=qk_scale,
            #         drop=drop_rate,
            #         attn_drop=attn_drop_rate,
            #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            #         norm_layer=norm_layer,
            #         norm_cfg=norm_cfg,
            #         downsample=OverlapPatchEmbed(3, 2, int(embed_dim * 2 ** i_layer),
            #                                      int(embed_dim * 2 ** (i_layer + 1)), norm_cfg=norm_cfg),
            #         use_checkpoint=use_checkpoint)
            # else:
            layer = CrossBasicLayer(dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                norm_cfg=norm_cfg,
                downsample=OverlapPatchEmbed(3, 2, int(embed_dim * 2 ** i_layer),
                                             int(embed_dim * 2 ** (i_layer + 1)), norm_cfg=norm_cfg) if i_layer < self.num_layers - 1 else None)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
                fan_out //= m.groups
                normal_init(
                    m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                # reshape = getattr(self, f'reshape{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2)  # B C H W
                outs.append(out)
        return tuple(outs)

    # def train(self, mode=True):
    #     super(CCrossTransformer).train(mode)
    #     self._freeze_stages()
