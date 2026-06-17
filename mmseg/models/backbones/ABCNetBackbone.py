import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter

from mmcv.cnn import build_norm_layer
from ..builder import BACKBONES
import re
from packaging import version

def l2_norm(x):
    # x: (B, C, N)
    return torch.einsum("bcn,bn->bcn", x, 1.0 / (torch.norm(x, p=2, dim=-2) + 1e-12))


def build_norm(norm_cfg, num_features, postfix=''):
    # build_norm_layer -> (name, layer)
    return build_norm_layer(norm_cfg, num_features, postfix=postfix)[1]


class Attention(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super().__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.eps = eps
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv   = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        Q = self.query_conv(x).view(b, -1, n)
        K = self.key_conv(x).view(b, -1, n)
        V = self.value_conv(x).view(b, -1, n)

        Q = l2_norm(Q).permute(0, 2, 1)   # (B, N, Cq)
        K = l2_norm(K)                    # (B, Cq, N)

        tailor_sum = 1.0 / (n + torch.einsum("bnc,bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1).expand(-1, c, n)
        matrix = torch.einsum("bmn,bcn->bmc", K, V)  # (B, Cq, C)
        matrix_sum = value_sum + torch.einsum("bnm,bmc->bcn", Q, matrix)
        weight_value = torch.einsum("bcn,bn->bcn", matrix_sum, tailor_sum).view(b, c, h, w)
        return (self.gamma * weight_value).contiguous()


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, norm_cfg, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn   = build_norm(norm_cfg, out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# -------- ResNet18 (你的实现，支持可配置 norm) --------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, norm_cfg=None):
        super().__init__()
        assert norm_cfg is not None, "norm_cfg must be provided"

        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1   = build_norm(norm_cfg, out_chan, postfix='1')
        self.conv2 = conv3x3(out_chan, out_chan, 1)
        self.bn2   = build_norm(norm_cfg, out_chan, postfix='2')
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                build_norm(norm_cfg, out_chan, postfix='d'),
            )

    def forward(self, x):
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(shortcut + residual)


def create_layer_basic(in_chan, out_chan, bnum, stride=1, norm_cfg=None):
    layers = [BasicBlock(in_chan, out_chan, stride=stride, norm_cfg=norm_cfg)]
    for _ in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1, norm_cfg=norm_cfg))
    return nn.Sequential(*layers)

def load_ckpt(ckpt_path):
    torch_version = torch.__version__
    if version.parse(torch_version) >= version.parse("2.0.0"):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"Loaded checkpoint with weights_only=True (PyTorch {torch_version})")
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"Loaded checkpoint without weights_only (PyTorch {torch_version})")
    return ckpt

class Resnet18(nn.Module):
    """输出 feat8/feat16/feat32"""
    def __init__(self, ckpt_path=None, norm_cfg=None):
        super().__init__()
        assert norm_cfg is not None, "norm_cfg must be provided"

        self.norm_cfg = norm_cfg
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = build_norm(norm_cfg, 64, postfix='1')
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1  = create_layer_basic(64, 64,  bnum=2, stride=1, norm_cfg=norm_cfg)
        self.layer2  = create_layer_basic(64, 128, bnum=2, stride=2, norm_cfg=norm_cfg)   # 1/8
        self.layer3  = create_layer_basic(128, 256, bnum=2, stride=2, norm_cfg=norm_cfg)  # 1/16
        self.layer4  = create_layer_basic(256, 512, bnum=2, stride=2, norm_cfg=norm_cfg)  # 1/32

        if ckpt_path is not None:
            self.load_pretrained(ckpt_path)

    def load_pretrained(self, ckpt_path, strict=False, logger=None):
        state_dict = load_ckpt(ckpt_path)
        filtered = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}

        model_dict = self.state_dict()
        load_dict = {}
        for k, v in filtered.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                load_dict[k] = v
            else:
                print(f"Skip loading parameter: {k}, required shape: {model_dict[k].shape}, loaded shape: {v.shape}")

        ret = self.load_state_dict(load_dict, strict=False)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        feat8  = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class AttentionEnhancementModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm_cfg):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, norm_cfg=norm_cfg, ks=3, stride=1, padding=1)
        self.att  = Attention(out_chan)
        self.bn   = build_norm(norm_cfg, out_chan, postfix='aem')

    def forward(self, x):
        feat = self.conv(x)
        att  = self.att(feat)
        return self.bn(att)


class ContextPath(nn.Module):
    """输出 feat_cp8(1/8,128) 和 feat_cp16(1/16,128)"""
    def __init__(self, resnet_ckpt=None, norm_cfg=None):
        super().__init__()
        assert norm_cfg is not None, "norm_cfg must be provided"
        self.resnet = Resnet18(ckpt_path=resnet_ckpt, norm_cfg=norm_cfg)

        self.arm16 = AttentionEnhancementModule(256, 128, norm_cfg=norm_cfg)
        self.arm32 = AttentionEnhancementModule(512, 128, norm_cfg=norm_cfg)

        self.conv_head32 = ConvBNReLU(128, 128, norm_cfg=norm_cfg, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, norm_cfg=norm_cfg, ks=3, stride=1, padding=1)
        self.conv_avg    = ConvBNReLU(512, 128, norm_cfg=norm_cfg, ks=1, stride=1, padding=0)

        self.up32 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False)
        self.up16 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat_cp16  = self.conv_head32(self.up32(feat32_sum))  # 1/16

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat_cp16
        feat_cp8   = self.conv_head16(self.up16(feat16_sum))  # 1/8

        return feat_cp8, feat_cp16


class SpatialPath(nn.Module):
    def __init__(self, norm_cfg=None):
        super().__init__()
        assert norm_cfg is not None, "norm_cfg must be provided"
        self.conv1 = ConvBNReLU(3,  64, norm_cfg=norm_cfg, ks=7, stride=2, padding=3)  # 1/2
        self.conv2 = ConvBNReLU(64, 64, norm_cfg=norm_cfg, ks=3, stride=2, padding=1)  # 1/4
        self.conv3 = ConvBNReLU(64, 64, norm_cfg=norm_cfg, ks=3, stride=2, padding=1)  # 1/8
        self.out   = ConvBNReLU(64, 128, norm_cfg=norm_cfg, ks=1, stride=1, padding=0) # 1/8

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out(x)
        return x


class FeatureAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm_cfg=None):
        super().__init__()
        assert norm_cfg is not None, "norm_cfg must be provided"
        self.convblk = ConvBNReLU(in_chan, out_chan, norm_cfg=norm_cfg, ks=1, stride=1, padding=0)
        self.att     = Attention(out_chan)

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)  # 256
        feat = self.convblk(fcat)
        atten = self.att(feat)
        return feat * atten + feat


@BACKBONES.register_module()
class ABCNetBackbone(nn.Module):
    """
    outs[0] = feat_fuse (1/8,256)
    outs[1] = feat_cp8  (1/8,128)
    outs[2] = feat_cp16 (1/16,128)
    """
    def __init__(self,
                 out_indices=(0, 1, 2),
                 resnet_ckpt=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super().__init__()
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        self.norm_cfg = norm_cfg

        self.cp  = ContextPath(resnet_ckpt=resnet_ckpt, norm_cfg=norm_cfg)
        self.sp  = SpatialPath(norm_cfg=norm_cfg)
        self.fam = FeatureAggregationModule(256, 256, norm_cfg=norm_cfg)

    def forward(self, x):
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.fam(feat_sp, feat_cp8)
        outs = [feat_fuse, feat_cp8, feat_cp16]
        return [outs[i] for i in self.out_indices]
