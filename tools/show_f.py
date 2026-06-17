# Copyright (c) OpenMMLab. All rights reserved.
# mmsegmentation==0.24 compatible
#
# 仅输出你要的结果（并明确“heatmap”也保存）：
# 1) pred_mask_idx.png + pred_mask_rgb.png
# 2) ALIGN: D-CAM / B-CAM（针对 --align-cls-id）
# 3) BACKBONE agnostic（你截图那种）：L2-norm response（彩色）+ overlay
#    -> AG_backbone_stage{i}_l2.png / AG_backbone_stage{i}_l2_overlay.png
# 4) BACKBONE heatmap（与 agnostic(L2) 数值一致，仅命名不同，方便你论文/对照）
#    -> stage{i}_heat.png / stage{i}_heat_overlay.png
# 5) BACKBONE cls1-4 “gram图”(按 Grad-CAM)：每个stage + overlay
#    -> CAM_backbone_stage{i}_cls{k}.png / CAM_backbone_stage{i}_cls{k}_overlay.png
#    可选保存灰度 CAM: CAM_backbone_stage{i}_cls{k}_u8.png（--save-cam-u8）
#
# 备注：
# - backbone per-class CAM 的目标分数：logits[cls] 在 pred==cls 区域的均值（如需 GT 区域可再改）
# - backbone agnostic(L2) 是 feature 的 L2 response map（不是 CAM）
#
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import shutil
import numpy as np
import cv2
import torch

from mmcv.parallel import collate, scatter
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose


# ===============================
# Hooks
# ===============================
class ActHook:
    """store activation and retain grad (for CAM)"""
    def __init__(self, need_grad=False):
        self.act = None
        self.need_grad = need_grad

    def __call__(self, module, inputs, outputs):
        self.act = outputs
        if self.need_grad and isinstance(outputs, torch.Tensor):
            outputs.retain_grad()


class BackboneListActHook:
    """store backbone outputs(list/tuple of tensors) and retain grad for each stage"""
    def __init__(self, need_grad=False):
        self.acts = None
        self.need_grad = need_grad

    def __call__(self, module, inputs, outputs):
        self.acts = outputs
        if not self.need_grad:
            return
        if isinstance(outputs, (list, tuple)):
            for t in outputs:
                if isinstance(t, torch.Tensor):
                    t.retain_grad()
        elif isinstance(outputs, torch.Tensor):
            outputs.retain_grad()


# ===============================
# Utils: visualize
# ===============================
def feature_to_heat_u8(feat_chw: torch.Tensor, out_hw=None):
    """
    feat_chw: [C,h,w] -> heat_u8: [H,W] uint8 in [0,255]
    """
    feat = feat_chw.float()
    heat = torch.norm(feat, p=2, dim=0)  # [h,w]
    heat -= heat.min()
    heat /= (heat.max() + 1e-6)
    heat_u8 = (heat.detach().cpu().numpy() * 255).astype(np.uint8)
    if out_hw is not None:
        heat_u8 = cv2.resize(heat_u8, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return heat_u8


def colorize_u8(u8_hw: np.ndarray):
    return cv2.applyColorMap(u8_hw, cv2.COLORMAP_JET)


def overlay(img_bgr, heat_bgr, alpha=0.45):
    heat_bgr = cv2.resize(heat_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
    return cv2.addWeighted(img_bgr, 1 - alpha, heat_bgr, alpha, 0)


def cam_to_color_overlay(img_bgr, cam_u8, alpha=0.45):
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    ov = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    return heat, ov


def mask_to_boundary(mask01: np.ndarray, k=3):
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(mask01.astype(np.uint8), kernel, iterations=1)
    ero = cv2.erode(mask01.astype(np.uint8), kernel, iterations=1)
    bd = (dil - ero) > 0
    return bd.astype(np.uint8)


# ===============================
# CAM (generic Grad-CAM)
# ===============================
def gradcam_from_act(act_bchw: torch.Tensor, out_hw):
    """
    act_bchw: [B,C,h,w], act_bchw.grad must exist
    return cam_u8: [H,W]
    """
    assert act_bchw is not None, "activation is None"
    assert act_bchw.grad is not None, "activation grad is None (backward not called?)"

    A = act_bchw[0]          # [C,h,w]
    G = act_bchw.grad[0]     # [C,h,w]

    w = G.mean(dim=(1, 2))   # [C]
    cam = torch.relu((w[:, None, None] * A).sum(dim=0))  # [h,w]
    cam -= cam.min()
    cam /= (cam.max() + 1e-6)

    cam_u8 = (cam.detach().cpu().numpy() * 255).astype(np.uint8)
    cam_u8 = cv2.resize(cam_u8, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return cam_u8


# ===============================
# Pred mask -> RGB
# ===============================
def get_palette(model, num_classes: int):
    palette = None
    if hasattr(model, "PALETTE") and model.PALETTE is not None:
        palette = model.PALETTE

    if palette is None:
        try:
            if hasattr(model, "cfg"):
                for split in ["train", "val", "test"]:
                    if split in model.cfg.data and "dataset" in model.cfg.data[split]:
                        ds = model.cfg.data[split]["dataset"]
                        if "palette" in ds:
                            palette = ds["palette"]
                            break
        except Exception:
            palette = None

    if palette is None:
        rng = np.random.RandomState(0)
        palette = rng.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
        palette[0] = np.array([0, 0, 0], dtype=np.uint8)

    palette = np.array(palette, dtype=np.uint8)
    if palette.shape[0] < num_classes:
        pad = np.zeros((num_classes - palette.shape[0], 3), dtype=np.uint8)
        palette = np.vstack([palette, pad])
    return palette[:num_classes]


def idx_to_rgb(mask_idx: np.ndarray, palette: np.ndarray):
    h, w = mask_idx.shape
    rgb = palette[mask_idx.reshape(-1)].reshape(h, w, 3)
    return rgb


# ===============================
# mmseg0.24 data -> logits with grad
# ===============================
def build_test_pipeline(cfg):
    return Compose(cfg.data.test.pipeline)


def _unwrap(x):
    if hasattr(x, "data"):  # DataContainer
        return x.data
    return x


def forward_logits_with_grad(model, img_path, device):
    cfg = model.cfg
    pipeline = build_test_pipeline(cfg)

    data = dict(img_info=dict(filename=img_path), img_prefix=None)
    data = pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if device.startswith("cuda"):
        data = scatter(data, [device])[0]

    # img tensor
    img_pack = _unwrap(data["img"])
    if isinstance(img_pack, list):
        img0 = _unwrap(img_pack[0])
        if isinstance(img0, list):
            img_tensor = img0[0]
        else:
            img_tensor = img0
    else:
        img_tensor = img_pack

    if isinstance(img_tensor, list):
        img_tensor = img_tensor[0]
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # [1,C,H,W]

    # img metas
    metas_pack = data.get("img_metas", None)
    if metas_pack is None:
        metas_pack = data.get("img_meta", None)
    if metas_pack is None:
        raise KeyError("data 中找不到 img_metas/img_meta")

    metas_pack = _unwrap(metas_pack)
    if isinstance(metas_pack, list):
        meta0 = _unwrap(metas_pack[0])
        if isinstance(meta0, list):
            img_metas = meta0
        else:
            img_metas = [meta0]
    else:
        img_metas = metas_pack

    # forward (不要 torch.no_grad)
    model.eval()
    logits = model.encode_decode(img_tensor, img_metas)  # [1,C,H,W]
    pred_idx = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    H, W = img_bgr.shape[:2]
    return logits, pred_idx, img_bgr, H, W


# ===============================
# BACKBONE agnostic(L2) + heatmap(同值不同名) + overlay
# ===============================
def dump_backbone_l2_maps(img_bgr, feats_list, outdir, alpha):
    """
    同时保存两套名字（数值相同）：
      - stage{i}_heat.png / stage{i}_heat_overlay.png
      - AG_backbone_stage{i}_l2.png / AG_backbone_stage{i}_l2_overlay.png
    """
    H, W = img_bgr.shape[:2]
    for i in range(4):
        f = feats_list[i]
        f_chw = f[0] if f.dim() == 4 else f  # [C,h,w]

        l2_u8 = feature_to_heat_u8(f_chw, out_hw=(H, W))
        l2_bgr = colorize_u8(l2_u8)
        ov = overlay(img_bgr, l2_bgr, alpha=alpha)

        # heatmap（你要的“heatmap”）
        cv2.imwrite(os.path.join(outdir, f"stage{i}_heat.png"), l2_bgr)
        cv2.imwrite(os.path.join(outdir, f"stage{i}_heat_overlay.png"), ov)


# ===============================
# BACKBONE per-class CAM (Grad-CAM) + overlay (+ optional u8)
# ===============================
def dump_backbone_class_cams(model, logits, pred_idx, img_bgr, feats_list,
                            outdir, cls_ids, alpha, min_pixels=10, save_cam_u8=False):
    """
    对每个 stage、每个 cls_id 输出：
      CAM_backbone_stage{i}_cls{k}.png
      CAM_backbone_stage{i}_cls{k}_overlay.png
    可选：
      CAM_backbone_stage{i}_cls{k}_u8.png  (灰度0-255)
    """
    H, W = img_bgr.shape[:2]
    num_classes = logits.shape[1]

    for cls_id in cls_ids:
        if cls_id < 0 or cls_id >= num_classes:
            print(f"[WARN] cls_id={cls_id} out of range, skip.")
            continue

        cls_mask = (pred_idx == cls_id).astype(np.uint8)
        pix = int(cls_mask.sum())
        if pix < min_pixels:
            print(f"[WARN] cls_id={cls_id} pixels too few({pix}), skip.")
            continue

        cls_bool = torch.from_numpy(cls_mask).to(logits.device).bool()

        for i in range(4):
            act = feats_list[i]
            if act.dim() == 3:
                act = act.unsqueeze(0)  # [1,C,h,w]

            model.zero_grad(set_to_none=True)
            if act.grad is not None:
                act.grad.zero_()

            score = logits[0, cls_id][cls_bool].mean()
            score.backward(retain_graph=True)

            cam_u8 = gradcam_from_act(act, out_hw=(H, W))
            heat, ov = cam_to_color_overlay(img_bgr, cam_u8, alpha=alpha)

            if save_cam_u8:
                cv2.imwrite(os.path.join(outdir, f"CAM_backbone_stage{i}_cls{cls_id}_u8.png"), cam_u8)
            cv2.imwrite(os.path.join(outdir, f"CAM_backbone_stage{i}_cls{cls_id}.png"), heat)
            cv2.imwrite(os.path.join(outdir, f"CAM_backbone_stage{i}_cls{cls_id}_overlay.png"), ov)


def _merge_cams_u8(cam_list_u8, merge="max"):
    """
    cam_list_u8: list of [H,W] uint8
    return merged_u8: [H,W] uint8
    """
    if len(cam_list_u8) == 0:
        return None
    cams = np.stack(cam_list_u8, axis=0).astype(np.float32)  # [K,H,W]
    if merge == "max":
        merged = cams.max(axis=0)
    elif merge == "sum":
        merged = cams.sum(axis=0)
    else:
        raise ValueError("merge must be 'max' or 'sum'")

    # normalize to 0..255
    merged -= merged.min()
    merged /= (merged.max() + 1e-6)
    merged_u8 = (merged * 255).astype(np.uint8)
    return merged_u8


def dump_align_dcam_bcam_multi(model, logits, pred_idx, img_bgr, align_act_bchw,
                                         outdir, cls_ids, bd_k, alpha,
                                         min_pixels=10, merge="max"):
    """
    在同一张图中融合多个类别的 ALIGN D-CAM / B-CAM。

    输出：
      DCAM_align_multi.png
      DCAM_align_multi_overlay.png
      BCAM_align_multi.png
      BCAM_align_multi_overlay.png
    """
    H, W = img_bgr.shape[:2]
    num_classes = logits.shape[1]

    if align_act_bchw is None:
        print("[WARN] ALIGN activation is None, skip.")
        return

    # -------------------------
    # D-CAM: per-class -> merge
    # -------------------------
    dcam_list = []
    valid_d = []

    for cls_id in cls_ids:
        if cls_id < 0 or cls_id >= num_classes:
            continue

        cls_mask = (pred_idx == cls_id).astype(np.uint8)
        if int(cls_mask.sum()) < min_pixels:
            continue

        cls_bool = torch.from_numpy(cls_mask).to(logits.device).bool()

        model.zero_grad(set_to_none=True)
        if align_act_bchw.grad is not None:
            align_act_bchw.grad.zero_()

        score_d = logits[0, cls_id][cls_bool].mean()
        score_d.backward(retain_graph=True)

        cam_d_u8 = gradcam_from_act(align_act_bchw, out_hw=(H, W))
        dcam_list.append(cam_d_u8)
        valid_d.append(cls_id)

    if len(dcam_list) == 0:
        print("[WARN] No valid classes for D-CAM (pixels too few).")
    else:
        merged_d_u8 = _merge_cams_u8(dcam_list, merge=merge)
        heat_d, ov_d = cam_to_color_overlay(img_bgr, merged_d_u8, alpha=alpha)

        tag = "_".join(map(str, valid_d))
        cv2.imwrite(os.path.join(outdir, f"DCAM_align_multi_{merge}_cls{tag}.png"), heat_d)
        cv2.imwrite(os.path.join(outdir, f"DCAM_align_multi_{merge}_cls{tag}_overlay.png"), ov_d)

    # -------------------------
    # B-CAM: per-class boundary -> merge
    # -------------------------
    bcam_list = []
    valid_b = []

    for cls_id in cls_ids:
        if cls_id < 0 or cls_id >= num_classes:
            continue

        cls_mask = (pred_idx == cls_id).astype(np.uint8)
        if int(cls_mask.sum()) < min_pixels:
            continue

        bd = mask_to_boundary(cls_mask, k=bd_k)
        if int(bd.sum()) < min_pixels:
            continue

        bd_bool = torch.from_numpy(bd).to(logits.device).bool()

        model.zero_grad(set_to_none=True)
        if align_act_bchw.grad is not None:
            align_act_bchw.grad.zero_()

        score_b = logits[0, cls_id][bd_bool].mean()
        score_b.backward(retain_graph=True)

        cam_b_u8 = gradcam_from_act(align_act_bchw, out_hw=(H, W))
        bcam_list.append(cam_b_u8)
        valid_b.append(cls_id)

    if len(bcam_list) == 0:
        print("[WARN] No valid classes for B-CAM (boundary too few).")
    else:
        merged_b_u8 = _merge_cams_u8(bcam_list, merge=merge)
        heat_b, ov_b = cam_to_color_overlay(img_bgr, merged_b_u8, alpha=alpha)

        tag = "_".join(map(str, valid_b))
        cv2.imwrite(os.path.join(outdir, f"BCAM_align_multi_{merge}_cls{tag}.png"), heat_b)
        cv2.imwrite(os.path.join(outdir, f"BCAM_align_multi_{merge}_cls{tag}_overlay.png"), ov_b)

    print("✅ ALIGN multi-class merged D/B-CAM saved (same image).")



# ===============================
# main
# ===============================
def parse_int_list(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='runs/vaihingen/others/sffnet/sffnet.512x512.vaihingen.160k.py')
    parser.add_argument('--checkpoint', default='runs/vaihingen/others/sffnet/best_mIoU_iter_21000.pth')
    parser.add_argument('--img', default='/home/dell/x/data/vaihingen/img_dir/val/area27_512_2048_1024_2560.png')
    parser.add_argument('--outdir', default='work_dirs/map/mfc/vis_backbone_align2')
    parser.add_argument('--device', default='cuda:0')

    # backbone per-class CAM ids (默认 1-4)
    parser.add_argument('--cls-ids', type=str, default='1,2,3,4',
                        help='backbone CAM class ids, e.g. "1,2,3,4"')

    # ALIGN CAM class (只做一个类的 D/B-CAM)
    parser.add_argument('--align-cls-id', type=str, default='1,2,3,4',
                        help='target class id for ALIGN D/B-CAM')

    parser.add_argument('--bd-k', type=int, default=3, help='boundary kernel size for B-CAM')
    parser.add_argument('--alpha', type=float, default=0.45, help='overlay alpha')
    parser.add_argument('--min-pixels', type=int, default=10, help='min pixels for a class region')
    parser.add_argument('--save-cam-u8', action='store_true', help='also save grayscale CAM u8 maps')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # copy original image into outdir
    img_name = os.path.basename(args.img)
    dst_img_path = os.path.join(args.outdir, img_name)
    if not os.path.exists(dst_img_path):
        shutil.copy(args.img, dst_img_path)

    # load model
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()

    # allow gradients (for CAM)
    for p in model.parameters():
        p.requires_grad_(True)

    # hooks: backbone + align only
    bb_act = BackboneListActHook(need_grad=True)
    h_bb = model.backbone.register_forward_hook(bb_act)

    # align_act = ActHook(need_grad=True)
    # h_align = model.decode_head.align.register_forward_hook(align_act)

    # forward (with grad)
    logits, pred_idx, img_bgr, H, W = forward_logits_with_grad(model, args.img, args.device)

    num_classes = logits.shape[1]
    palette = get_palette(model, num_classes)
    pred_rgb = idx_to_rgb(pred_idx, palette)      # RGB
    pred_bgr = pred_rgb[..., ::-1].copy()         # BGR for cv2
    cv2.imwrite(os.path.join(args.outdir, "pred_mask_rgb.png"), pred_bgr)

    # backbone outputs
    feats = bb_act.acts
    if feats is None or (not isinstance(feats, (list, tuple))) or len(feats) < 4:
        raise RuntimeError(f"❌ backbone 输出异常: {type(feats)}, len={len(feats) if isinstance(feats,(list,tuple)) else 'NA'}")

    # 2) backbone heatmap + agnostic(L2)
    dump_backbone_l2_maps(img_bgr, feats, args.outdir, alpha=args.alpha)
    h_bb.remove()

    print("🎉 Finished: pred + ALIGN(D/B) + backbone(heat+agnostic L2 + cls1-4 Grad-CAM).")


if __name__ == "__main__":
    main()
