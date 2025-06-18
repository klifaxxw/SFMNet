import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmseg.models import build_segmentor

import torch
from torchprofile import profile_macs

from mmseg.models.utils import selective_scan_flop_jit
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count, parameter_count_table

def parse_args():
    parser = argparse.ArgumentParser(description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    args = parser.parse_args()
    return args

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batched_inputs):
        img_meta = {
            'filename': 'D:/x/model/model/unet/VOCdevkit/VOC2007/JPEGImages/0.png',
            'flip': False,
            'flip_direction': 'horizontal',
            'img_norm_cfg': {
                'mean': [0.0, 0.0, 0.0],
                'std': [255.0, 255.0, 255.0],
                'to_rgb': True
            },
            'img_shape': (512, 512, 3),
            'ori_filename': '0.png',
            'ori_shape': (512, 512, 3),
            'pad_shape': (512, 512, 3),
            'scale_factor': 1.0
        }
        images = (batched_inputs / 255.0).cuda()  # [C, H, W]
        # images = torch.stack(images, dim=0)  # [N, C, H, W]
        # 如果你的模型需要输入归一化或其他处理，可以放在这里
        logits = self.model([images], [[img_meta]], return_loss=False)  # [N, C, H, W]
        return [{"sem_seg": torch.from_numpy(x)} for x in logits]  # 返回 List[Dict[str, Tensor]]


def get_flops(model, input_shape=(1, 3, 512, 512)):
    model = WrappedModel(model)
    supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,  # latter
        }
    shape = (3, 512, 512)
    input = torch.randn((1, *shape), device=next(model.parameters()).device)
    # params = parameter_count(model)[""]
    Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
    print(parameter_count_table(model))
    print(f"GFLOPs {sum(Gflops.values())}")

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()
    get_flops(model)


if __name__ == '__main__':
    main()
