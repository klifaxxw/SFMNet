import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm
from timm.models.layers import DropPath, trunc_normal_

from mmcv.cnn import build_norm_layer
from ..builder import BACKBONES
from packaging import version
import re

def build_norm(norm_cfg, num_features, postfix=''):
    # mmcv build_norm_layer returns (name, layer)
    return build_norm_layer(norm_cfg, num_features, postfix=postfix)[1]


def norm_layer_factory(norm_cfg):
    # return a callable: norm_layer(channels) -> norm module
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


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


# ====== 关键：修正你原始 SeparableConvBN/BNReLU 的 BN 通道不匹配问题 ======
# 原代码 depthwise 后写 norm_layer(out_channels) 在一般情况下会 shape mismatch。
# 这里是可泛化且可跑的 separable conv：DW(in)->BN(in)->PW(out)->BN(out)
class SeparableConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, dilation=dilation,
            padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            groups=in_channels, bias=False
        )
        self.bn1 = norm_layer(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class SeparableConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, dilation=dilation,
            padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            groups=in_channels, bias=False
        )
        self.bn1 = norm_layer(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.pw(x)
        x = self.bn2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8,
                 relative_pos_embedding=True, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3, norm_layer=norm_layer)
        self.local2 = ConvBN(dim, dim, kernel_size=1, norm_layer=norm_layer)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size, norm_layer=norm_layer)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,
                                   padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1,
                                   padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding
        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, ws, ws
            coords_flatten = torch.flatten(coords, 1)  # 2, ws*ws
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, ws*ws, ws*ws
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # ws*ws, ws*ws
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(
            qkv,
            'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            h=self.num_heads, d=C // self.num_heads,
            hh=Hp // self.ws, ww=Wp // self.ws,
            qkv=3, ws1=self.ws, ws2=self.ws
        )

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(
            attn,
            '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads, d=C // self.num_heads,
            hh=Hp // self.ws, ww=Wp // self.ws,
            ws1=self.ws, ws2=self.ws
        )
        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]
        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         window_size=window_size, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3, norm_layer=norm_layer)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3, norm_layer=norm_layer)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ReLU6(),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1, norm_layer=norm_layer)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3, norm_layer=norm_layer)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x

        x = self.post_conv(x)
        shortcut = self.shortcut(x)

        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca

        x = self.proj(x) + shortcut
        x = self.act(x)
        return x

def load_ckpt(ckpt_path):
    torch_version = torch.__version__
    if version.parse(torch_version) >= version.parse("2.0.0"):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"Loaded checkpoint with weights_only=True (PyTorch {torch_version})")
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"Loaded checkpoint without weights_only (PyTorch {torch_version})")
    return ckpt

@BACKBONES.register_module()
class UNetFormerBackbone(nn.Module):
    def __init__(self,
                 backbone_name='swsl_resnet18',
                 output_stride=32,
                 out_indices=(1, 2, 3, 4),
                 decode_channels=64,
                 window_size=8,
                 pretrained=False,
                 pretrained_ckpt="weights/rs3_mamba_pretrained.pth",
                 norm_cfg=dict(type='BN', requires_grad=True),
                 convert_syncbn=False):
        super().__init__()

        self.encoder = timm.create_model(backbone_name, features_only=True, output_stride=output_stride,
                                          out_indices=out_indices, pretrained=pretrained)
        pretrained_weight = load_ckpt(pretrained_ckpt)
        self.encoder.load_state_dict(pretrained_weight, strict=False)
        if convert_syncbn:
            self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

        encoder_channels = self.encoder.feature_info.channels()  # e.g. [64,128,256,512]
        c1, c2, c3, c4 = encoder_channels

        nl = norm_layer_factory(norm_cfg)

        self.pre_conv = ConvBN(c4, decode_channels, kernel_size=1, norm_layer=nl)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size, norm_layer=nl)

        self.p3 = WF(c3, decode_channels, norm_layer=nl)
        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size, norm_layer=nl)

        self.p2 = WF(c2, decode_channels, norm_layer=nl)
        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size, norm_layer=nl)

        self.p1 = FeatureRefinementHead(c1, decode_channels, norm_layer=nl)

        # 这三个在原 decoder 里是 if self.training: 才创建
        # 但你要 backbone “始终返回 h4,h3,h2”，所以这里统一创建（不影响推理，只是多了几层）
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        
    
 

    def forward(self, x):
        # timm outputs: res1,res2,res3,res4
        res1, res2, res3, res4 = self.encoder(x)

        # ====== 原 decoder 中 segmentation_head/aux_head 之前的步骤 ======
        y = self.b4(self.pre_conv(res4))
        h4 = self.up4(y)

        y = self.p3(y, res3)
        y = self.b3(y)
        h3 = self.up3(y)

        y = self.p2(y, res2)
        y = self.b2(y)
        h2 = y

        x_feat = self.p1(y, res1)  # feature before segmentation_head
        x_aux = h4 + h3 + h2  # auxiliary feature
        return [x_feat, x_aux]
