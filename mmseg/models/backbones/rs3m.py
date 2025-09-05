from mmcv.runner import BaseModule
from ..utils.rs3_layer import *
from collections import OrderedDict
from ..builder import BACKBONES
import timm
import torch
import re
from packaging import version

# Code adapted from RS3_mamba: https://github.com/sstary/SSRS.git
# mamba forward_type: differnt cuda version get different compiled version.
# when cuda version > 12.1, dont have selective_scan_cuda_core

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
class RS3MBackbone(BaseModule):
    def __init__(self,
                 in_chans=3,
                 dims=[96, 192, 384, 768],
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 pretrained_ckpt="weights/rs3_mamba_pretrained.pth",
                 norm_layer="sb",
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        # self.conv1 = self.backbone.conv1
        # self.bn1 = self.backbone.bn1
        # self.act1 = self.backbone.act1
        # self.maxpool = self.backbone.maxpool
        # self.layers = nn.ModuleList()
        # self.layers.append(self.backbone.layer1)
        # self.layers.append(self.backbone.layer2)
        # self.layers.append(self.backbone.layer3)
        # self.layers.append(self.backbone.layer4)
        _NORMLAYERS = dict(ln=nn.LayerNorm, bn=nn.BatchNorm2d, sb=nn.SyncBatchNorm)
        norm_layer = _NORMLAYERS[norm_layer.lower()]
        self.stem = nn.Sequential(
              nn.Conv2d(in_chans, 48, kernel_size=7, stride=2, padding=3),
              nn.InstanceNorm2d(48, eps=1e-5, affine=True),
        )
        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=48, dims=dims)
        encoder_channels = self.backbone.feature_info.channels()
        # self.Fuse = nn.ModuleList()
        # for i in range(4):
        #     fuse = FusionBlock(encoder_channels[i], dims[i], norm_layer=norm_layer)
        #     self.Fuse.append(fuse)
        self.Fuse = nn.ModuleList([FusionBlock(encoder_channels[i], dims[i], norm_layer=norm_layer) for i in range(4)])
        self.load_pretrained_ckpt(pretrained_ckpt)


    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def load_pretrained_ckpt(self, 
        ckpt_path=None):
        if ckpt_path is None:
            return
        skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias", 
                    "patch_embed.proj.weight", "patch_embed.proj.bias", 
                    "patch_embed.norm.weight", "patch_embed.norm.weight"]    
        ckpt = load_ckpt(ckpt_path)
        model_dict = self.state_dict()
        for k, v in ckpt['model'].items():
            if k in skip_params:
                continue
            kr = f"vssm_encoder.{k}"
            if "downsample" in kr:
                i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
                kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
                assert kr in model_dict.keys()
            if kr in model_dict.keys():
                assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
                model_dict[kr] = v
        self.load_state_dict(model_dict)


    def forward(self, x):
        h, w = x.size()[-2:]
        ssmx = self.stem(x)
        vss_outs = self.vssm_encoder(ssmx) # 48*128*128, 96*64*64, 192*32*32, 384*16*16, 768*8*8
        ress = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        for i in range(4):
            layer = getattr(self.backbone, f"layer{i+1}", nn.Identity())
            x = layer(x)
            x = self.Fuse[i](x, vss_outs[i+1])
            res = x
            ress.append(res)
        return ress

# if __name__ == "__main__":
#     model = RS3MBackbone().cuda()
#     img = torch.randn(1, 3, 512, 512).cuda()
#     outs = model(img)
#     for i in range(len(outs)):
#         print(i, outs[i].shape)
    