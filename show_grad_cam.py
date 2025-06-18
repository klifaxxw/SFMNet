from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
import torch.nn.functional as F
from mmseg.apis import init_segmentor, show_result_pyplot
from pytorch_grad_cam import GradCAM, ShapleyCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import os
import shutil

class WrappedModel(nn.Module):
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
        images = batched_inputs.cuda()  # [N, C, H, W]
        logits = self.model([images], [[img_meta]], return_loss=False)  # [N, C, H, W]
        return logits


class SemanticSegmentationTarget:
    """Wrap the model for Grad-CAM.

    Args:
        category (int): Visualization class.
        mask (ndarray): Mask of class.
        size (tuple): Image size.
    """

    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = F.interpolate(
            model_output, size=self.size, mode='bilinear'
        )
        model_output = torch.squeeze(model_output, dim=0)
        return (model_output[self.category, :, :] * self.mask).sum()


if __name__ == '__main__':
    # 配置文件和模型权重文件
    config = r"Umamba_tiny/unet_mamba_tiny_soild_160k_adwn.py"
    model_state_file = r"Umamba_tiny/best_mIoU_iter_152000.pth"
    model = init_segmentor(config, model_state_file)

    # 输入图像路径
    img_path = r"/home/dell/x/model/unet/VOCdevkit/VOC2007/JPEGImages/33.png"
    category = 1
    basename = os.path.basename(img_path)

    name = os.path.splitext(basename)[0]
    save_dir = f"{name}/grad_cam{category}" 
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # 包装模型
    model = WrappedModel(model).cuda()

    # 目标层列表（多个层）
    target_layers = [
        ["model.model.backbone.reshape1", model.model.backbone.reshape1],
        ["model.model.backbone.reshape2", model.model.backbone.reshape2],
        ["model.model.backbone.reshape3", model.model.backbone.reshape3],
        ["model.model.backbone.reshape4", model.model.backbone.reshape4],
        ["model.model.backbone.reshape5", model.model.backbone.reshape5],
        ["model.model.decode_head.reshape_final", model.model.decode_head.reshape_final],
    ]

    # 加载输入图像
    shutil.copy(img_path, os.path.join(save_dir,"o_" + basename))
    oimg = Image.open(img_path)
    rimg = oimg.convert('RGB')
    image = np.array(rimg)
    height, width = image.shape[0], image.shape[1]
    rgb_img = np.float32(image) / 255

    # 输入图像预处理
    image_mean = [0, 0, 0]  # 根据模型配置文件调整
    image_std = [255, 255, 255]
    input_tensor = preprocess_image(
        rgb_img,
        mean=[x / 255 for x in image_mean],
        std=[x / 255 for x in image_std]
    )
    # 推理，获取预测结果
    seg_logit = model(input_tensor)
    seg_mask = seg_logit.argmax(dim=1).detach().cpu().numpy().squeeze(0)
    drimg = np.array(oimg, dtype=np.uint8)
    drimg[seg_mask == 1] = [119, 191, 179]
    Image.fromarray(drimg).save(os.path.join(save_dir, basename))
    # Image.fromarray(seg_mask.astype(np.uint8)).save(f"{save_dir}/{basename}")
    mask_c1 = np.float32(seg_mask == category)

    # 定义 Grad-CAM 目标
    targets = [
        SemanticSegmentationTarget(category, mask_c1, (height, width))
    ]

    # 生成并保存多层 Grad-CAM 结果
    for idx, d in enumerate(target_layers):
        layer_path, target_layer = d
        layer_name = target_layer.__class__.__name__  # 获取层的名字
        with GradCAM(
                model=model,
                target_layers=[target_layer]
        ) as cam:
            # 生成 Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # 保存可视化结果
            output_file = f"{save_dir}/grad_cam_{layer_path.replace('.', '_')}.jpg"
            Image.fromarray(cam_image).save(output_file)
            print(f"Grad-CAM saved for layer {layer_name} to {output_file}")

        with ShapleyCAM(
                model=model,
                target_layers=[target_layer]
        ) as cam:
            # ShapleyCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # 保存可视化结果
            output_file = f"{save_dir}/shapley_cam_{layer_path.replace('.', '_')}.jpg"
            Image.fromarray(cam_image).save(output_file)
            print(f"Shapley-CAM saved for layer {layer_name} to {output_file}")

    
        with EigenGradCAM(
            model=model,
            target_layers=[target_layer]
        ) as cam:
            # ShapleyCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # 保存可视化结果
            output_file = f"{save_dir}/eigen_cam_{layer_path.replace('.', '_')}.jpg"
            Image.fromarray(cam_image).save(output_file)
            print(f"eigen-CAM saved for layer {layer_name} to {output_file}")
