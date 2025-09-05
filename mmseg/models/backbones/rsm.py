from mmcv.runner import BaseModule
from ..utils.rs_layer import *
from collections import OrderedDict
from ..builder import BACKBONES
# Code adapted from RS_mamba: https://github.com/walking-shadow/Official_Remote_Sensing_Mamba.git
# mamba forward_type: differnt cuda version get different compiled version.
# when cuda version > 12.1, dont have selective_scan_cuda_core

@BACKBONES.register_module()
class RSMBackbone(BaseModule):
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.1,
                 patch_norm=True,
                 norm_layer="LN",
                 use_checkpoint=False,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer="silu",
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_drop_rate=0.0,
                 ssm_init="v0",
                 forward_type="v2",
                 mlp_ratio=4.0,
                 mlp_act_layer="gelu",
                 mlp_drop_rate=0.0,
                 init_cfg=None,
                
                 **kwargs):
        super().__init__(init_cfg)

        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.dims = dims
        self.num_features = dims[-1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # norm & act dict
        _NORMLAYERS = dict(ln=nn.LayerNorm, bn=nn.BatchNorm2d, sn=nn.SyncBatchNorm)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU)

        if isinstance(norm_layer, str):
            norm_layer = _NORMLAYERS[norm_layer.lower()]
        if isinstance(ssm_act_layer, str):
            ssm_act_layer = _ACTLAYERS[ssm_act_layer.lower()]
        if isinstance(mlp_act_layer, str):
            mlp_act_layer = _ACTLAYERS[mlp_act_layer.lower()]

        # patch embedding
        self.patch_embed = self._make_patch_embed_v2(
            in_chans, dims[0], patch_size, patch_norm, norm_layer)

        # encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = (self._make_downsample_v3(
                self.dims[i_layer - 1], self.dims[i_layer], norm_layer
            ) if i_layer != 0 else nn.Identity())

            self.encoder_layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    def _make_layer(self, dim=96, drop_path=[0.1, 0.1], **kwargs):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(OSSBlock(hidden_dim=dim, drop_path=drop_path[d], **kwargs))
        return nn.Sequential(OrderedDict(
            downsample=kwargs.get("downsample", nn.Identity()),
            blocks=nn.Sequential(*blocks)
        ))

    def forward(self, x):
        outs = []
        x = self.patch_embed(x)
        for layer in self.encoder_layers:
            x = layer(x)
            outs.append(x)
        # return multi-scale feature maps
        return [rearrange(f, "b h w c -> b c h w") for f in outs]

if __name__ == "__main__":
    model = RSMBackbone().cuda()
    img = torch.randn(1, 3, 512, 512).cuda()
    outs = model(img)
    for i in range(len(outs)):
        print(i, outs[i].shape)
