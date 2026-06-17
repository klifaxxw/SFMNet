import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from pytorch_wavelets import DWTForward, DWTInverse
import torch.fft as fft


class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'),
                 e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(c1, c_, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv3 = ConvModule(c_, c_, 3, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv4 = ConvModule(c_, c_, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = ConvModule(4 * c_, c_, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv6 = ConvModule(c_, c_, 3, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv7 = ConvModule(2 * c_, c2, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class Sb(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.bias, mean=mean, std=std)

    def forward(self, x):
        return x + self.bias


class Sw(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.weight, mean=mean, std=std)

    def forward(self, x):
        return x * self.weight

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
            k = torch.empty(kernel, kernel).uniform_(-1, 1)
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

class FCExchange(nn.Module):
    def __init__(self, in_channels, level=1, align_corners=False,
                 conv_cfg=None, norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'), wave='haar', J=1):
        super().__init__()
        assert len(in_channels) >= 2, "must have at least two feature levels"

        self.in_channels = in_channels
        self.level = level
        self.align_corners = align_corners
        self.dwt = DWTForward(J=J, mode='symmetric', wave=wave)
        self.idwt = DWTInverse(wave=wave, mode='zero')
        # 反向通道列表
        in_channels_inv = in_channels[::-1]
        # 高频特征路径
        self.high_exchange_blocks = nn.ModuleList([
            nn.Sequential(
                ConvModule(in_channels_inv[i], in_channels_inv[i], 1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                SFConv(in_channels_inv[i], kernel=5, mode="random"),
                ConvModule(in_channels_inv[i], in_channels_inv[i + 1], 1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
            for i in range(len(in_channels_inv) - 1)
        ])

        self.high_converters = nn.ModuleList([
            ConvModule(in_channels_inv[i + 1] * 2, in_channels_inv[i + 1], 1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for i in range(len(in_channels_inv) - 1)
        ])

        # 低频特征路径
        self.low_exchange_blocks = nn.ModuleList([
            nn.Sequential(
                ConvModule(in_channels[i], in_channels[i], 1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                SFConv(in_channels[i], kernel=3, mode="random"),
                ConvModule(in_channels[i], in_channels[i + 1], 1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
            for i in range(len(in_channels) - 1)
        ])

        self.low_converters = nn.ModuleList([
            ConvModule(in_channels[i + 1] * 2, in_channels[i + 1], 1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for i in range(len(in_channels) - 1)
        ])

    def forward(self, inputs):
        # 统一分辨率
        target_size = inputs[self.level].shape[2:]
        inputs = [F.interpolate(feat, size=target_size, mode='bilinear',
                                align_corners=self.align_corners)
                  for feat in inputs]
        inputs = inputs[::-1]
        for i in range(len(self.high_exchange_blocks)):
            x1 = inputs[i]
            x2 = inputs[i + 1]

            y1, _ = self.dwt(x1)
            y1 = self.high_exchange_blocks[i](y1)

            y2, yh2 = self.dwt(x2)
            y2 = self.high_converters[i](torch.cat([y2, y1], dim=1))
            inputs[i + 1] = self.idwt((y2, yh2))
        inputs = inputs[::-1]
        for i in range(len(self.low_exchange_blocks)):
            x1 = inputs[i]
            x2 = inputs[i + 1]
            _, yh_list1 = self.dwt(x1)
            hh1 = yh_list1[0][:, :, 2]
            hh1 = self.low_exchange_blocks[i](hh1)

            y2, yh_list2 = self.dwt(x2)
            hh2 = yh_list2[0][:, :, 2]
            hh2 = self.low_converters[i](torch.cat([hh2, hh1], dim=1))
            yh_list2[0][:, :, 2] = hh2
            inputs[i + 1] = self.idwt((y2, yh_list2))
        return inputs


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)
        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=None,
            act_cfg=None
        )

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


@HEADS.register_module()
class LightHamHead(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 bn_norm_cfg=None,
                 **kwargs):
        super(LightHamHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels
        self.sppcsp = SPPCSPC(self.in_channels[-1], self.in_channels[-1], norm_cfg=bn_norm_cfg)
        self.FCE = FCExchange(self.in_channels, level=1, align_corners=self.align_corners, norm_cfg=bn_norm_cfg)
        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        inputs[-1] = self.sppcsp(inputs[-1])
        # inputs = [resize(
        #     level,
        #     size=inputs[1].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners
        # ) for level in inputs]
        inputs = self.FCE(inputs)
        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)
        x = self.hamburger(x)
        output = self.align(x)
        output = self.cls_seg(output)
        return output
