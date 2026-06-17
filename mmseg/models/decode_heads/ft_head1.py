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

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()

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


class Gate(nn.Module):
    def __init__(self, c_in, reduction=8):
        super().__init__()
        mid = max(c_in // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(c_in, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c_in)
        )
    def forward(self, feature1, feature2):
        g = torch.mean(feature2, dim=[2, 3])
        gate = torch.sigmoid(self.fc(g)).view(g.size(0), -1, 1, 1)
        return feature1 + gate * feature1

class FeaturesFusion(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channel,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))
        assert len(in_channels) == 4, f'Expect 4 feature maps, got {len(in_channels)}'
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.conv_LG = ConvModule(
                sum(in_channels[2:]), mid_channel, 1, 1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None)

        self.conv_HL = ConvModule(
                sum(in_channels[1:3]), mid_channel, 1, 1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None)


        self.conv_HG = ConvModule(
                sum(in_channels[:2]), mid_channel, 1, 1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None)
            
        self.fconv_1 = nn.Sequential(
                ConvModule(
                2 * mid_channel, mid_channel, 1, 1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
                SFConv(mid_channel, weight=True, bias=True, shortcut=False, mode="ones"),
                ConvModule(
                    mid_channel, mid_channel, 1, 1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            )
        
        self.fconv_2 = nn.Sequential(
                ConvModule(
                2 * mid_channel, mid_channel, 1, 1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
                SFConv(mid_channel, weight=True, bias=True, shortcut=False, mode="ones"),
                ConvModule(
                    mid_channel, mid_channel, 1, 1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
        )
        self.local_gate = Gate(mid_channel)
        self.global_gate = Gate(mid_channel)
        self.fine = ConvModule(
            2 * mid_channel, out_channel, 3, 1, 1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


    def forward(self, feats):
        assert len(feats) == len(self.in_channels), \
            f"Expected {len(self.in_channels)} features, got {len(feats)}."
        x1, x2, x3, x4 = feats
        # Local feature and shadom semantic data
        HG = self.conv_HG(torch.cat([x1, x2], dim=1))
        # mid global feature and mid deep semantic data
        HL = self.conv_HL(torch.cat([x2, x3], dim=1))
        # global feature and deep semantic data
        LG = self.conv_LG(torch.cat([x3, x4], dim=1))
        local_feature = self.fconv_1(torch.cat([HG, HL], dim=1))
        global_feature = self.fconv_2(torch.cat([LG, HL], dim=1))
        local_feature = self.local_gate(local_feature, global_feature)
        global_feature = self.global_gate(global_feature, local_feature)
        x = torch.cat([local_feature, global_feature], dim=1)
        x = self.fine(x)
        return x


class FCExchange(nn.Module):
    def __init__(self, in_channels, level=1, align_corners=False,
                 conv_cfg=None,
                 act_cfg=dict(type='GELU'), norm_cfg=None, wave='haar', J=1):
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
        
        self.low_exchange_blocks = nn.ModuleList([
            nn.Sequential(
                ConvModule(in_channels_inv[i], in_channels_inv[i], 1,
                           conv_cfg=conv_cfg, norm_cfg=None, act_cfg=act_cfg),
                SFConv(in_channels_inv[i], kernel=5, shortcut=True, weight=True, bias=True, mode="ones"),
                ConvModule(in_channels_inv[i], in_channels_inv[i + 1], 1,
                           conv_cfg=conv_cfg, norm_cfg=None, act_cfg=act_cfg)
            )
            for i in range(len(in_channels_inv) - 1)
        ])

        self.low_converters = nn.ModuleList([
            ConvModule(in_channels_inv[i + 1] * 2, in_channels_inv[i + 1], 1,
                       conv_cfg=conv_cfg, norm_cfg=None, act_cfg=act_cfg)
            for i in range(len(in_channels_inv) - 1)
        ])

        # 低频特征路径
        self.high_exchange_blocks = nn.ModuleList([
            nn.Sequential(
                ConvModule(in_channels[i], in_channels[i], 1,
                           conv_cfg=conv_cfg, norm_cfg=None, act_cfg=act_cfg),
                SFConv(in_channels[i], kernel=3, shortcut=True, weight=True, bias=True, mode="random"),
                ConvModule(in_channels[i], in_channels[i + 1], 1,
                           conv_cfg=conv_cfg, norm_cfg=None, act_cfg=act_cfg)
            )
            for i in range(len(in_channels) - 1)
        ])
        self.high_converters = nn.ModuleList([
            ConvModule(in_channels[i + 1] * 2, in_channels[i + 1], 1,
                       conv_cfg=conv_cfg, norm_cfg=None, act_cfg=act_cfg)
            for i in range(len(in_channels) - 1)
        ])
        self.norm = nn.ModuleList([
            LayerNorm2d(ch) for ch in in_channels
        ])
        
    def forward(self, inputs):
        target_size = inputs[self.level].shape[2:]
        inputs = [F.interpolate(feat, size=target_size, mode='bilinear',
                                align_corners=self.align_corners)
                  for feat in inputs]
        features = [f.clone() for f in inputs]
        inputs = inputs[::-1]
        for i in range(len(self.low_exchange_blocks)):
            x1 = inputs[i]
            x2 = inputs[i + 1]
            y1, _ = self.dwt(x1)
            y1 = self.low_exchange_blocks[i](y1)
            y2, yh2_list = self.dwt(x2)
            y2 = self.low_converters[i](torch.cat([y2, y1], dim=1))
            inputs[i + 1] = self.idwt((y2, yh2_list))
        inputs = inputs[::-1]
        for i in range(len(self.high_exchange_blocks)):
            x1 = inputs[i]
            x2 = inputs[i + 1]
            _, yh_list1 = self.dwt(x1)
            hh1 = yh_list1[0].permute(0, 2, 1, 3, 4)  # b c, 3, h w -> b 3, c, h, w
            b, _, c, h, w = hh1.shape
            hh1 = hh1.reshape(b * 3, c, h, w)
            hh1 = self.high_exchange_blocks[i](hh1)
            y2, yh_list2 = self.dwt(x2)
            hh2 = yh_list2[0].permute(0, 2, 1, 3, 4)
            _, _, c1, _, _ = hh2.shape
            hh2 = hh2.reshape(b * 3, c1, h, w)
            hh2 = self.high_converters[i](torch.cat([hh2, hh1], dim=1))
            hh2 = hh2.reshape(b, 3, c1, h, w).permute(0, 2, 1, 3, 4)
            inputs[i + 1] = self.idwt((y2, [hh2]))
        
        for i, (s, f) in enumerate(zip(features, inputs)):
            norm = self.norm[i]
            inputs[i] = norm(f + s)
    
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
class HLFT1(BaseDecodeHead):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 bn_norm_cfg=None,
                 **kwargs):
        super(HLFT1, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels
        self.sppcsp = SPPCSPC(self.in_channels[-1], self.in_channels[-1], norm_cfg=bn_norm_cfg)
        self.FCE = FCExchange(self.in_channels, level=1, align_corners=self.align_corners, norm_cfg=bn_norm_cfg)
        
        # self.fuse = FeaturesFusion(self.in_channels, self.ham_channels // 2, self.ham_channels,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)

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
        inputs = self.FCE(inputs)
        # x = self.fuse(inputs)
        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)
        x = self.hamburger(x)
        output = self.align(x)
        output = self.cls_seg(output)
        return output
