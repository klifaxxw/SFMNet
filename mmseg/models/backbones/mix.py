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
from mmseg.utils.logger import get_root_logger
from ..utils.crss_layer import *
from mmseg.core import MSDeformAttn


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sizes = sizes

    def forward(self, x):
        y = x.permute((0, 3, 1, 2))
        return y




class depthwise(nn.Module):
    def __init__(self, embed_dim, kernel_size, norm_cfg=dict(type='LN', requires_grad=True)):
        super(depthwise, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, kernel_size), padding=autopad((1, kernel_size)), groups=embed_dim)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=(kernel_size, 1), padding=autopad((kernel_size, 1)), groups=embed_dim)
        # self.bn = build_norm_layer(norm_cfg, embed_dim)[1]
        # self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        # x = self.bn(x)
        # x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        # x = self.act(x)
        return x
        

class FEModule(nn.Module):
    def __init__(self, embed_dim, kernels=None, norm_cfg=dict(type='LN', requires_grad=True)):
        super(FEModule, self).__init__()
        assert kernels is not None and isinstance(kernels, list), "kernels must be a list of integers"
        self.num_conv = len(kernels)
        self.dc = nn.ModuleList()
        for i, kernel_size in enumerate(kernels):
            self.dc.append(depthwise(embed_dim, kernel_size, norm_cfg))
    
    def forward(self, x):
        outs = []
        outs.append(x)
        for layer in self.dc:
            out = layer(x)
            outs.append(out)
        return outs


class AttentionModule(nn.Module):
    def __init__(self, embed_dim, n_heads, n_points=4, kernels=None, norm_cfg=dict(type='LN', requires_grad=True)):
        super(AttentionModule, self).__init__()
        self.convert = nn.Conv2d(embed_dim, embed_dim, 1)
        self.num_features = len(kernels) + 1  # include the original feature
        self.extract_features = FEModule(embed_dim, kernels=kernels, norm_cfg=norm_cfg)
        self.deformAttn = MSDeformAttn(embed_dim, self.num_features, n_heads, n_points)
        # self.level_embed = nn.Parameter(torch.Tensor(self.num_features, embed_dim))
        self.fuse = nn.Conv2d(self.num_features * embed_dim, embed_dim, 1)


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        # bs, 2
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio



    def get_reference_points(self, spatial_shape, valid_ratio, device):
        H_, W_ = spatial_shape
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        # (1, h * w) / bs, None,
        ref_y = ref_y.reshape(-1)[None] / (valid_ratio[:, None, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratio[:, None, 0] * W_)
        # bs, h_ * w_, 2
        reference_points = torch.stack((ref_x, ref_y), -1)
        # (bs, h_ * w_, 2) * (bs, 1, 2)
        reference_points = reference_points * valid_ratio[:, None]
        # (bs, h_ * w_, 2)
        return reference_points
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    
    def forward(self, x, pos_enc):
        bs, c, h, w = x.shape
        spatial_shape = [[h, w]] * self.num_features
        mask = torch.zeros_like(x[:, 0, :, :], dtype=torch.bool, device=x.device)
        # bs, 2 当使用数据增强时，考虑有效mask
        valid_ratio = self.get_valid_ratio(mask)
        # bs, h * w, 2
        reference_points = self.get_reference_points(spatial_shape[0], valid_ratio, x.device).repeat(1, self.num_features, 1)
        # mul local features
        mul_features = self.extract_features(x)
        src = torch.stack(mul_features, dim=2).flatten(3).flatten(2).transpose(1, 2).contiguous()
        spatial_shape = torch.tensor(spatial_shape, dtype=torch.int32, device=x.device)
        attn = self.deformAttn(self.with_pos_embed(src, pos_enc), reference_points, src, spatial_shape)
        # fuse
        # attn = attn.reshape((bs, self.num_features, h * w, -1)).transpose(1, 2).reshape((bs, h * w, -1))
        x = attn.reshape((bs, self.num_features, h * w, -1)).transpose(2, 3).reshape((bs, -1, h, w))
        attn = self.fuse(x)
        return attn


class MixAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_points=4, ssm_ratio=1, global_mode=True, kernels=None, norm_cfg=dict(type='LN', requires_grad=True)):
        super(MixAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn = AttentionModule(embed_dim, num_heads, num_points, kernels=kernels, norm_cfg=norm_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        self.ssm = VSSM(d_model=embed_dim, expansion_ratio=ssm_ratio)
        self.dwconv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
                nn.BatchNorm2d(embed_dim),
                )
        self.act = nn.GELU()
        self.norm1 = build_norm_layer(norm_cfg, embed_dim)[1]
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    
    ### atten 运算
    def forward(self, x, pos_enc):
        b, c, h, w = x.shape
        shorcut = x
        atten = self.attn(x, pos_enc)
        atten = atten.flatten(2).transpose(1, 2).contiguous()  # (b, c, h, w) -> (b, h * w, c)
        x = x + self.norm(atten).reshape((b, h, w, c)).permute(0, 3, 1, 2)  # (b, h * w, c) -> (b, c, h, w)
        v_r = x
        v = self.act(self.dwconv(v_r))
        v = self.ssm(v)
        v = v.permute(0, 2, 1)
        v = self.norm1(v)
        v = v.permute(0, 2, 1).reshape((b, -1, h, w))
        x = v_r + v
        x = x + shorcut
        x = self.proj(x)
        return x


class MixBlock(nn.Module): 
    def __init__(self, embed_dim, num_heads, num_points=4, mlp_ratio=1.0, drop_path=0, global_mode=False, kernels=[7, 11, 21], is_scale=False, layer_init_values=1e-6, norm_cfg=None, ssm_ratio=1):
        super(MixBlock, self).__init__()
        self.embed_dim = embed_dim
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm1 = LayerNorm2d(embed_dim)
        self.attn = MixAttention(embed_dim, num_heads, num_points, ssm_ratio=ssm_ratio, global_mode=global_mode, kernels=kernels, norm_cfg=norm_cfg)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm2 = LayerNorm2d(embed_dim)
        self.mlp = FFN(embed_dim, int(embed_dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if is_scale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x, pos_enc):
        x = x + self.cpe1(x)
        x = x + self.drop_path(self.layer_scale1(self.attn(self.norm1(x), pos_enc)))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        return x


class MixLayer(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, kernels=[7, 11, 21], num_points=4, mlp_ratio=4, drop_path=0, is_scale=True, layer_scale_init_values=1e-6, downsample=None, norm_cfg=None,
    use_checkpoint=0, pos_func="relative", ssm_ratio=1):
        assert norm_cfg is not None, "norm_cfg not be None"
        super(MixLayer, self).__init__()
        if pos_func == "relative":
            self.encode_pos = PositionEmbeddingSine(embed_dim // 2, normalize=True)
        else:
            raise ValueError(f"Unsupported positional encoding function: {pos_func}")
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        self.num_features = len(kernels) + 1  # include the original feature
        # self.level_embed = nn.Parameter(torch.Tensor(self.num_features, embed_dim))
        self.level_embed = nn.Embedding(self.num_features, embed_dim)
        for i in range(depths):
            block = MixBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_points=num_points,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                kernels=kernels,
                is_scale=is_scale,
                layer_init_values=layer_scale_init_values,
                norm_cfg=norm_cfg,
                ssm_ratio=ssm_ratio,
                global_mode = False
            )
            self.blocks.append(block)
        if downsample is not None:
            self.downsample = downsample(embed_dim, norm_cfg=norm_cfg)


    def forward(self, x):
        # x: (N, C, H, W)
        poss = []
        pos = self.encode_pos(x)
        pos = pos.flatten(2).transpose(1, 2).contiguous()
        for i in range(self.num_features):
            l_pos = self.level_embed.weight[i].view(1, 1, -1).contiguous()
            pos = pos + l_pos
            poss.append(pos)
        # bs, h1 * w1 + h2 * w2 + ... + h_feature_n * w_feature_n, c
        pos = torch.cat(poss, dim=1)
        for i, blk in enumerate(self.blocks):
            if i < self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        # x: (N, C, H / 2, W / 2)
        if hasattr(self, 'downsample'):
            x = self.downsample(x)
        return x


@BACKBONES.register_module()
class mixer(BaseModule):
    def __init__(self,
                 in_channels=3,
                 depths=[2, 2, 9, 4],
                 dims=[64, 144, 288, 512],
                 num_heads=[2, 4, 8, 16],
                 mlp_ratios=[4, 4, 4, 4],
                 kernels=[7, 11, 21],
                 num_points=[4, 6, 6, 8],
                 drop_path_rate=0.1,
                 layer_init_values=1e-6,
                 norm_cfg=dict(type='LN', requires_grad=True),
                 patch_norm= dict(type='SyncBN', requires_grad=True),
                 use_checkpoint=False,
                 out_indices=(0, 1, 2, 3),
                 pos_func="relative", # positional encoding function: rope\relative\absolute
                 ssm_ratio=1,
                 scale=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        assert len(depths) == len(dims) == len(num_heads) == len(num_points) == len(mlp_ratios), "depths, dims, num_heads, num_points and mlp_ratios must have the same length"
        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.out_indices = out_indices
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # 提取特征
        self.patch_embed0 = StemConv(in_channels, dims[0], patch_norm)
        # self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MixLayer(
                embed_dim=dims[i_layer],
                depths=depths[i_layer],
                num_heads=num_heads[i_layer],
                kernels=kernels,
                num_points=num_points[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                is_scale=scale,
                layer_scale_init_values=layer_init_values,
                downsample=None,
                norm_cfg=norm_cfg,
                use_checkpoint=use_checkpoint,
                pos_func=pos_func,
                ssm_ratio=ssm_ratio
            )
            layer_downsample = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=dims[i_layer], embed_dim=dims[i_layer + 1], norm_cfg=patch_norm) if i_layer < self.num_layers - 1 else None
            norm = build_norm_layer(norm_cfg, dims[i_layer])[1]
            setattr(self, f"patch_embed{i_layer + 1}", layer_downsample)
            setattr(self, f"norm{i_layer + 1}", norm)
            setattr(self, f"layer{i_layer + 1}", layer)
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
        outs = []
        for i_layer in range(self.num_layers):
            patch_embed = getattr(self, f"patch_embed{i_layer}")
            layer = getattr(self, f"layer{i_layer + 1}")
            norm = getattr(self, f"norm{i_layer + 1}")
            # bs, c, h, w
            x = patch_embed(x)
            x = layer(x)
            b, c, h, w = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = norm(x)
            x = x.transpose(1, 2).contiguous().reshape((b, c, h, w))
            if i_layer in self.out_indices:
                outs.append(x)
        return outs