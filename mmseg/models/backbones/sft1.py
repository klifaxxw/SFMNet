import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
from mmseg.models.builder import BACKBONES

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from pytorch_wavelets import DWTForward, DWTInverse
import torch.fft as fft


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.down = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

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

        attn_3 = self.up(self.down(attn))

        attn = attn + attn_0 + attn_1 + attn_2 + attn_3
        attn = self.conv3(attn)
        return attn * u

class Sb(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(Sb, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class Sw(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(Sw, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

class SFConv(nn.Module):
    def __init__(self, channels, kernel=3, weight=False, bias=False, shortcut=False, mode="random"):
        super().__init__()
        self.channels = channels
        self.shortcut = shortcut

        if weight:
            self.sw = Sw(channels)
        else:
            self.sw = nn.Identity()

        if bias:
            self.sb = Sb(channels)
        else:
            self.sb = nn.Identity()

        if shortcut:
            self.shortcut_proj = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)

        # 初始化频率滤波核
        if mode == "random":
            k = torch.empty(kernel, kernel).uniform_(-0.5, 0.5)
        elif mode == "ones":
            k = torch.ones(kernel, kernel)
        elif mode == "zeros":
            k = torch.zeros(kernel, kernel)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.kernel = nn.Parameter(k)

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        # 频域变换
        F_fft = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        # 构造核
        kernel_padded = torch.zeros((h, w), dtype=x.dtype, device=device)
        kh, kw = self.kernel.shape
        kernel_padded[:kh, :kw] = self.kernel
        FX = fft.fftshift(fft.fft2(kernel_padded))
        FX = FX.unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
        # 频率乘积 & 逆变换
        G = F_fft * FX
        G = torch.abs(fft.ifft2(fft.ifftshift(G, dim=(-2, -1)), dim=(-2, -1)))

        # 权重与偏置调制
        G = self.sb(self.sw(G))

        if self.shortcut:
            return self.shortcut_proj(G) + x
        else:
            return G


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()

# kernels: [5, 9, 15]
class SFourierKTrans(nn.Module):
    def __init__(self, d_model, kernel_sizes=[3, 5, 9], mode="random"):
        super().__init__()
        self.proj = nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model)
        self.kernel_sizes = kernel_sizes
        self.mode = mode
        # 注册可学习参数卷积核
        for size in kernel_sizes:
            if mode == "random":
                k = torch.empty(size, size).uniform_(-0.5, 0.5)
            elif mode == "ones":
                k = torch.ones(size, size)
            elif mode == "zeros":
                k = torch.zeros(size, size)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            param = nn.Parameter(k)
            setattr(self, f"sw_{size}", Sw(d_model))
            setattr(self, f"sb_{size}", Sb(d_model))
            setattr(self, f"kernel_{size}", param)
        self.proj1 = nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        r = None
        F_fft = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        for size in self.kernel_sizes:
            kernel = getattr(self, f"kernel_{size}")
            sw = getattr(self, f"sw_{size}")
            sb = getattr(self, f"sb_{size}")
            kernel_p = torch.zeros((h, w), dtype=x.dtype, device=device)
            kh, kw = kernel.shape 
            ph, pw = min(kh, h), min(kw, w)
            kernel_p[:ph, :pw] = kernel[:ph, :pw]
            # kernel_p[:size, :size] = kernel
            FX = fft.fftshift(fft.fft2(kernel_p))
            FX = FX.unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
            G = F_fft * FX
            G = torch.abs(fft.ifft2(fft.ifftshift(G, dim=(-2, -1)), dim=(-2, -1)))
            G = sb(sw(G))
            if r is None:
                r = G
            else:
                r = r + G
        r = self.proj1(r)
        x = r + x
        return x


class SFourierHighTrans(nn.Module):
    def __init__(self, mode="fine"):
        super().__init__()
        sobelh = torch.tensor([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]],
                              dtype=torch.float32)
        sobelv = torch.tensor([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]],
                              dtype=torch.float32)
        self.register_buffer('sobelh', sobelh)
        self.register_buffer('sobelv', sobelv)
        if mode == "fine":
            self.sobelh_b = nn.Parameter(torch.zeros(3, 3, dtype=torch.float32))
            self.sobelv_b = nn.Parameter(torch.zeros(3, 3, dtype=torch.float32))
        elif mode == "constant":
            self.sobelh_b = None
            self.sobelv_b = None
        else:
            raise NotImplementedError(f"Mode {mode} is not supported for SFourierHighTrans")

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        if getattr(self, 'sobelh_b', None) is not None:
            sobelh = self.sobelh + self.sobelh_b
            sobelv = self.sobelv + self.sobelv_b
        else:
            sobelh = self.sobelh
            sobelv = self.sobelv
        fx = torch.zeros((h, w), dtype=x.dtype, device=device)
        fy = torch.zeros((h, w), dtype=x.dtype, device=device)
        fx[:3, :3] = sobelh
        fy[:3, :3] = sobelv
        FX = fft.fftshift(fft.fft2(fx))
        FY = fft.fftshift(fft.fft2(fy))
        FX = FX.view(1, 1, h, w)
        FY = FY.view(1, 1, h, w)
        G = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        Gx = G * FX
        Gy = G * FY
        gx = torch.abs(fft.ifft2(fft.ifftshift(Gx, dim=(-2, -1)), dim=(-2, -1)))
        gy = torch.abs(fft.ifft2(fft.ifftshift(Gy, dim=(-2, -1)), dim=(-2, -1)))
        g = torch.sqrt(gx ** 2 + gy ** 2)
        return g

class SFourierGaussLowHigh(nn.Module):
    def __init__(self, mode="fine", ks=5, sigma=1.0):
        super().__init__()
        assert ks % 2 == 1, "kernel size ks should be odd."
        self.ks = ks
        self.sigma = sigma
        gauss = self._gaussian_kernel_2d(ks, sigma)
        self.register_buffer("gauss", gauss)
        if mode == "fine":
            self.gauss_b = nn.Parameter(torch.zeros_like(gauss))
        elif mode == "constant":
            self.gauss_b = None
        else:
            raise NotImplementedError(f"Mode {mode} is not supported for SFourierGaussLowHigh")

    @staticmethod
    def _gaussian_kernel_2d(ks, sigma, dtype=torch.float32, device="cpu"):
        ax = torch.arange(ks, dtype=dtype, device=device) - ks // 2
        try:
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        except TypeError:
            xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        b, c, h, w = x.shape
        if getattr(self, "gauss_b", None) is not None:
            k = self.gauss.to(device=device, dtype=dtype) + self.gauss_b.to(device=device, dtype=dtype)
        else:
            k = self.gauss.to(device=device, dtype=dtype)
        k = k / (k.sum() + 1e-6)
        f = torch.zeros((h, w), dtype=dtype, device=device)
        ks = self.ks
        f[:ks, :ks] = k
        F = fft.fftshift(fft.fft2(f))   
        F = F.view(1, 1, h, w)
        G = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1)) 
        G = G * F
        G = torch.abs(fft.ifft2(fft.ifftshift(G, dim=(-2, -1)), dim=(-2, -1)))
        return G



class WTrans(nn.Module):
    def __init__(self, d_model, wave='haar', J=1):
        super().__init__()
        self.proj = nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model)
        self.down = nn.Conv2d(d_model, d_model, 3, 2, 1)
        self.act = nn.GELU()
        self.dwt = DWTForward(J=J, mode='symmetric', wave=wave)
        self.idwt = DWTInverse(wave=wave, mode='zero')
        self.proj2 = nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model)


    def forward(self, x, gx=None, gy=None):
        x = self.proj(x)
        yl, yh_list = self.dwt(x)
        yh = yh_list[0]  # 第一层高频 [B,C,H/2,W/2,3]
        lh, hl, hh = yh[:, :, 0], yh[:, :, 1], yh[:, :, 2]
        if gx is not None and gy is not None:
            lh_add = F.avg_pool2d(gx, kernel_size=2)
            hl_add = F.avg_pool2d(gy, kernel_size=2)
            hh_add = torch.sqrt(lh_add ** 2 + hl_add ** 2)
            y1_add = self.act(self.down(x))
            # 叠加能量幅度到原始高频
            lh = lh + lh_add
            hl = hl + hl_add
            hh = hh + hh_add
            yl = yl + y1_add

        yh = torch.stack([lh, hl, hh], dim=2)
        x = self.idwt((yl, [yh]))
        x = self.proj2(x)
        return x


class FrequencyAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model)
        self.sfft = SFourierHighTrans()
        self.wave = WTrans(d_model)
        self.proj1 = nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model)

    def forward(self, x):
        x1 = self.proj(x)
        gx, gy = self.sfft(x1)
        g = torch.sqrt(gx ** 2 + gy ** 2)
        x1 = self.wave(x1, gx=gx, gy=gy)
        x = x + self.proj1(x1)
        return x


class SpatialAttention(BaseModule):
    def __init__(self, d_model, kernels=[5, 9, 15], norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.d_model = d_model
        # self.convert_1 = nn.Conv2d(self.d_model, self.d_model, 1)
        # self.sk = SFourierKTrans(self.d_model, kernel_sizes=kernels)
        # self.norm = LayerNorm2d(self.d_model)
        # self.convert_2 = nn.Conv2d(self.d_model, self.d_model, 1)

        self.proj_1 = nn.Conv2d(self.d_model, self.d_model, 1)
        self.activation = nn.GELU()
        
        self.spatial_gating_unit = AttentionModule(self.d_model)
        self.proj_2 = nn.Conv2d(self.d_model, self.d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut

        # shortcut = x.clone()
        # x = self.convert_1(x)
        # x = self.sk(x)
        # x = self.norm(x)
        # x = self.convert_2(x)
        # x = x + shortcut
        return x


class Block(BaseModule):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 kernels=[5, 9, 15],
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim, norm_cfg=norm_cfg, kernels=kernels)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x

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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


@BACKBONES.register_module()
class SFTran1(BaseModule):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 kernels=[5, 9, 15],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        print(kernels)
        super(SFTran1, self).__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)
            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg, kernels=kernels)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(SFTran, self).init_weights()

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x):
        x = self.dwconv(x)
        return x