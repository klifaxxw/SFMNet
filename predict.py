import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch

import cv2
from PIL import Image
import torch.nn.functional as F
import os
from tqdm import tqdm
from skimage import morphology
import shutil
from torch.utils.data.dataset import Dataset
from torchinfo import summary
from thop import profile, clever_format
from register_soild_semantic import register_soild_semantic
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from mmcv.utils import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmseg.datasets.soild import Soild
from mmseg.models.utils import selective_scan_flop_jit

import time

class Timer:
    def __init__(self):
        self.total_time = 0.0
        self.count = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        self.total_time += elapsed
        self.count += 1

    def fps(self):
        if self.total_time == 0:
            return float('inf')
        return self.count / self.total_time



class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#
        image_path = os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".png")
        label_path = os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png")
        return image_path, label_path, name




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
        images = [x["image"].to(torch.device("cuda")).float() / 255.0 for x in batched_inputs]  # [C, H, W]
        images = torch.stack(images, dim=0)  # [N, C, H, W]
        # 如果你的模型需要输入归一化或其他处理，可以放在这里
        logits = self.model([images], [[img_meta]], return_loss=False)  # [N, C, H, W]
        return [{"sem_seg": torch.from_numpy(x)} for x in logits]  # 返回 List[Dict[str, Tensor]]

class WrappedModel1(torch.nn.Module):
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

mean = [0, 0, 0]
std = [1, 1, 1]
colors = [ (0, 0, 0), (119, 191, 179), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count, parameter_count_table


def get_flops(model, input_shape=(1, 3, 512, 512)):
    model = WrappedModel1(model)
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


if __name__ == "__main__":

    # register_soild_semantic()
    # # dir_save_seg_path = r"Umamba/seg"
    # # dir_save_boundary_path = r"Umamba/boundary"
    # # Path(dir_save_seg_path).mkdir(parents=True, exist_ok=True)
    # # Path(dir_save_boundary_path).mkdir(parents=True, exist_ok=True)
    # config = r"unet_mamba_tiny_soild_160k_adwn.py"
    # cfg = Config.fromfile(config)
    # model_state_file = r"Umamba_tiny/iter_170000.pth"
    # cfg.model.train_cfg = None
    # model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    # model.cuda().eval()
    # get_flops(model)
    
    # # register_soild_semantic()
    # checkpoint = load_checkpoint(model, model_state_file, map_location='cpu')
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model.CLASSES = checkpoint['meta']['CLASSES']
    # model.PALETTE = checkpoint['meta']['PALETTE']
    # distributed = False
    # dataset = build_dataset(cfg.data.train)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False)
    # ## 第二部分
    # efficient_test = False
    # model = model.cuda().eval()
    # timer = Timer()
    # timer1 = Timer()
    # for data in tqdm(dataset):
    #     gt = data['gt_semantic_seg'].data
    #     img = data['img'].data[None]
    #     img = img.cuda()
    #     meta = data['img_metas'].data
    #     img_path = meta["filename"]
    #     name = meta['ori_filename']
    #     newdata = {}
    #     metas = [[meta]]
    #     imgs = [img]
    #     with timer:
    #         result = model(img=imgs, img_metas=metas, rescale=False,return_loss=False)[0]
    #     pr = result.astype(np.uint8)
    #     seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [512, 512, -1])
    #     seg_image = Image.fromarray(np.uint8(seg_img))
    #     seg_image.save(os.path.join(dir_save_seg_path, name))
    #     with timer1:    
    #         imgs = morphology.remove_small_holes(pr, 50, connectivity=2)
    #         imgs = morphology.remove_small_objects(imgs, 50, connectivity=2)
    #         imgs = np.array(imgs, dtype=np.uint8)
    #         contours, hierarchys = cv2.findContours(imgs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     labelss = []
    #     labelso = []
    #     oimage = cv2.imread(img_path)
    #     if len(contours) > 0:
    #         for i, (contour, hierarchy) in enumerate(zip(contours, hierarchys[0])):
    #             child_idx = hierarchy[2]
    #             parent_idx = hierarchy[3]
    #             if parent_idx != -1:
    #                 labelso.append(contour.reshape((-1, 2)).tolist())
    #             else:
    #                 labelss.append(contour.reshape((-1, 2)).tolist())
    #         for contour in labelss:
    #             contour = np.array(contour)
    #             cv2.drawContours(oimage, [contour], -1, (0, 0, 255), 2)
    #         for contour in labelso:
    #             contour = np.array(contour)
    #             cv2.drawContours(oimage, [contour], -1, (255, 0, 0), 2)
    #     cv2.imwrite(os.path.join(dir_save_boundary_path, name), oimage)
    # print(f"Processed {timer.count} frames in {timer.total_time:.2f} seconds. FPS: {timer.fps():.2f}")
    # print(f"Processed {timer1.count} frames in {timer1.total_time:.2f} seconds. FPS: {timer1.fps():.2f}")
    # model = WrappedModel(model)
    # cfg = get_cfg()
    # cfg.DATASETS.TEST = "voc_soild_val"
    # cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.INPUT.MIN_SIZE_TEST = 512
    # cfg.INPUT.MAX_SIZE_TEST = 512
    # cfg.INPUT.FORMAT = "RGB"
    # cfg.TEST.AUG.ENABLED = False
    # cfg.TEST.BATCH_SIZE_PER_IMAGE = 1
    # # 7. 设置归一化均值和标准差（注意VOC原始数据一般是RGB格式）
    # # 这里均值、标准差可以根据训练时的计算结果调整，这里是ImageNet的常用值示范
    # cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
    # cfg.MODEL.PIXEL_STD = [1, 1, 1]
    # val_loader = build_detection_test_loader(cfg, "voc_soild_val")
    # evaluator = SemSegEvaluator(dataset_name="voc_soild_val")
    # metrics = inference_on_dataset(model, val_loader, evaluator)
    # print(metrics)
    #
    # outputs = single_gpu_test(model, data_loader, True,  "out",
    #                           efficient_test)



    # VOCdevkit_path = r'D:\x\model\model\unet\VOCdevkit'
    # dir_save_seg_path = r"F:\model_result\result\HRNet\seg"
    # dir_save_boundary_path = r"F:\model_result\result\HRNet\boundary"
    # Path(dir_save_seg_path).mkdir(parents=True, exist_ok=True)
    # Path(dir_save_boundary_path).mkdir(parents=True, exist_ok=True)
    # input_shape = (512, 512)
    # num_classes = 2
    # with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
    #     val_lines = f.readlines()
    # dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    # for data in tqdm(dataset):
    #     # img = input('Input image filename:')
    #     image_path, label_path, name = data
    #     oimg = Image.open(image_path)
    #     oimg = oimg.convert("RGB")
    #     oimage = np.asarray(oimg, dtype='uint8')
    #     oimg = torch.from_numpy(oimage).float() / 255.0
    #     oimg = oimg.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    #     pr = inference(config, model, oimg)
    #     bimage = pr.astype(np.uint8)
    #     seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [512, 512, -1])
    #     oimage = cv2.cvtColor(oimage, cv2.COLOR_RGB2BGR)
    #     r_image = Image.fromarray(np.uint8(seg_img))
    #     r_image.save(os.path.join(dir_save_seg_path, name + ".png"))
    #     imgs = morphology.remove_small_holes(bimage, 200, connectivity=2)
    #     imgs = morphology.remove_small_objects(imgs, 200, connectivity=2)
    #     imgs = np.array(imgs, dtype=np.uint8)
    #     contours, hierarchys = cv2.findContours(imgs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     labelss = []
    #     labelso = []
    #     if len(contours) > 0:
    #         for i, (contour, hierarchy) in enumerate(zip(contours, hierarchys[0])):
    #             child_idx = hierarchy[2]
    #             parent_idx = hierarchy[3]
    #             if parent_idx != -1:
    #                 labelso.append(contour.reshape((-1, 2)).tolist())
    #             else:
    #                 labelss.append(contour.reshape((-1, 2)).tolist())
    #         for contour in labelss:
    #             contour = np.array(contour)
    #             cv2.drawContours(oimage, [contour], -1, (0, 0, 255), 5)
    #     cv2.imwrite(os.path.join(dir_save_boundary_path, name + ".png"), oimage)

    # with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
    #     val_lines = f.readlines()
    # dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    # for data in tqdm(dataset):
    #     # img = input('Input image filename:')
    #     image_path, label_path, name = data
    #     oimg = Image.open(image_path)
    #     oimg = oimg.convert("RGB")
    #     oimage = np.asarray(oimg, dtype='uint8')
    #     oimg = torch.from_numpy(oimage).float() / 255.0
    #     oimg = oimg.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    #     pr = inference(config, model, oimg)
    #     bimage = pr.astype(np.uint8)
    #     seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [512, 512, -1])
    #     oimage = cv2.cvtColor(oimage, cv2.COLOR_RGB2BGR)
    #     r_image = Image.fromarray(np.uint8(seg_img))
    #     r_image.save(os.path.join(dir_save_seg_path, name + ".png"))
    #     imgs = morphology.remove_small_holes(bimage, 200, connectivity=2)
    #     imgs = morphology.remove_small_objects(imgs, 200, connectivity=2)
    #     imgs = np.array(imgs, dtype=np.uint8)
    #     contours, hierarchys = cv2.findContours(imgs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     labelss = []
    #     labelso = []
    #     if len(contours) > 0:
    #         for i, (contour, hierarchy) in enumerate(zip(contours, hierarchys[0])):
    #             child_idx = hierarchy[2]
    #             parent_idx = hierarchy[3]
    #             if parent_idx != -1:
    #                 labelso.append(contour.reshape((-1, 2)).tolist())
    #             else:
    #                 labelss.append(contour.reshape((-1, 2)).tolist())
    #         for contour in labelss:
    #             contour = np.array(contour)
    #             cv2.drawContours(oimage, [contour], -1, (0, 0, 255), 5)
    #     cv2.imwrite(os.path.join(dir_save_boundary_path, name + ".png"), oimage)


