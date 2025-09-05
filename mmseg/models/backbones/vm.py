from mmcv.runner import BaseModule
from ..utils.vm_layer import *
from collections import OrderedDict
from ..builder import BACKBONES
import timm
import torch
import re
from packaging import version

# Code adapted from RS3_mamba: https://github.com/JCruan519/VM-UNet.git
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
class VMUBackbone(BaseModule):
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768],
                 d_state=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 pretrained_ckpt="weights/vm_unet_tiny_e292.pth",
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.dims = dims
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
        self.apply(self._init_weights)
        self.load_pretrained_ckpt(pretrained_ckpt)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def load_pretrained_ckpt(self, ckpt_path=None):
        if ckpt_path is None:
            return
        ckpt = load_ckpt(ckpt_path)
        model_dict = self.state_dict()
        pretrained_dict = ckpt['model']
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        outs = []
        for i, layer in enumerate(self.layers):
            outs.append(x)
            x = layer(x)
        outs.append(x)
        return tuple(outs)

# from ..decode_heads.vm_head import VMUHead
# if __name__ == "__main__":
#     # ====== 构建模型 ======
#     model = VMUBackbone().cuda()
#     model1 = VMUHead(num_classes=6).cuda()
#     # ====== 构造输入 ======
#     dummy = torch.randn(1, 3, 512, 512).cuda()
#     # ====== 前向计算 ======
#     outputs = model(dummy)
#     # ====== 打印结果 ======
#     print("Number of feature maps:", len(outputs))
#     for i, feat in enumerate(outputs):
#         print(f"Feature[{i}] shape: {feat.shape}")
    
#     out = model1(outputs)
#     print("Output shape:", out.shape)
        
