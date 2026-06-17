# mmseg_backbone.py
from ..builder import BACKBONES
import timm
import torch.nn as nn
from mmcv.runner import BaseModule
import torch

@BACKBONES.register_module()
class sff(BaseModule):
    """
    sffnet backbone
    """
    def __init__(self,
                 model_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=False,
                 pretrained_ckpt=None,
                 out_indices=(0,1,2,3),
                 init_cfg=None,
                 **kwargs):
        super(sff, self).__init__(init_cfg)
        self.backbone = timm.create_model(model_name=model_name, features_only=True,pretrained=pretrained, output_stride=32, out_indices=out_indices)
        if pretrained_ckpt is not None and pretrained_ckpt != "":
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')
            new_state_dict = {}

            for k, v in state_dict.items():
                # 替换 stem.0 -> stem_0, stem.1 -> stem_1
                k_new = k.replace('stem.', 'stem_').replace('stages.', 'stages_')
                new_state_dict[k_new] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)


    def forward(self, x):
        feats = self.backbone(x)  # expect list/tuple
        return feats