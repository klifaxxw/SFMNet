from ..builder import HEADS
import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ..utils.vssm_layer import VSSLayer_up, PatchExpand, FinalPatchExpand_X4
import math
from timm.models.layers import  trunc_normal_

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sizes = sizes

    def forward(self, x):
        y = x.permute((0, 3, 1, 2))
        return y

@HEADS.register_module()
class MixHead(BaseDecodeHead):
    """Mamba UNet Decode Head"""
    def __init__(self,
                 depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768],
                 d_state=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 num_classes = 2,
                 in_index=[0, 1, 2, 3],
                 in_channels=[96, 192, 384, 768],
                 channels=128,
                 dropout_ratio=0.1,
                 norm_cfg=None,
                 align_corners=False,
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         input_transform='multiple_select',
                         in_channels=in_channels,
                         channels=channels,
                         in_index=in_index,
                        **kwargs)

        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.dims = dims
        self.num_classes = num_classes
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建解码器层
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(dims[0] * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(dims[0] * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(dim=int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2,
                                       norm_layer=norm_layer)
            else:
                layer_up = VSSLayer_up(
                    dim=int(dims[0] * 2 ** (self.num_layers - 1 - i_layer)),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        # 最终输出层
        self.norm_up = norm_layer(self.embed_dim)
        self.up = FinalPatchExpand_X4(dim_scale=4, dim=self.embed_dim)
        self.output = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.num_classes, kernel_size=1,
                                    bias=False)
        self.reshape_final = Reshape()  # 用于将输出转换为B,C,H,W格式
        self.reshape_final_1 = Reshape()
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'bias') and getattr(m.bias, '_no_reinit', False):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_up_features(self, x, x_downsample):
        """解码器前向传播"""
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B H W C
        return x

    def up_x4(self, x):
        """最终上采样x4"""
        B, H, W, C = x.shape
        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B C H W
        return x

    def forward(self, inputs):
        """实现BaseDecodeHead的forward接口"""
        x = inputs[-1].permute((0, 2, 3, 1))
        x_downsample = [z.permute((0, 2, 3, 1)) for z in inputs[:-1]]
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B H W C
        x = self.reshape_final(x)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = self.reshape_final_1(x)  # B,C,H,W
        x = self.output(x)
        return x