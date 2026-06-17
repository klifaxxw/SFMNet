# Copyright (c) OpenMMLab. All rights reserved.
# mmsegmentation==0.24 compatible
#
# 功能：
# 1) dump backbone 4-stage feature heatmap + overlay
# 2) dump FCE 4 outputs heatmap + overlay
# 3) dump fuse / hamburger / align feature heatmap + overlay
# 4) 在 ALIGN 特征上做 D-CAM / B-CAM（类似你给的 B-CAM / D-CAM 图）
# 5) 保存 pred_mask_idx.png + pred_mask_rgb.png（用 PALETTE / dataset palette）

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import shutil
import numpy as np
import cv2
import torch

import mmcv
from mmcv.parallel import collate, scatter
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose


# ===============================
# Hooks
# ===============================
class FeatureHook:
    def __init__(self):
        self.feats = None
    def __call__(self, module, inputs, outputs):
        self.feats = outputs


class ActHook:
    """store activation and retain grad (for CAM)"""
    def __init__(self, need_grad=False):
        self.act = None
        self.need_grad = need_grad
    def __call__(self, module, inputs, outputs):
        self.act = outputs
        if self.need_grad and isinstance(outputs, torch.Tensor):
            outputs.retain_grad()


# ===============================
# Utils: visualize
# ===============================
def feature_to_heatmap(feat_chw: torch.Tensor, out_hw=None):
    """
    feat_chw: [C,H,W]
    out_hw: (H,W)
    """
    feat = feat_chw.float()
    heat = torch.norm(feat, p=2, dim=0)  # [H,W]
    heat -= heat.min()
    heat /= (heat.max() + 1e-6)
    heat = (heat.detach().cpu().numpy() * 255).astype(np.uint8)
    if out_hw is not None:
        heat = cv2.resize(heat, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heat


def overlay(img_bgr, heat_bgr, alpha=0.45):
    heat_bgr = cv2.resize(heat_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
    return cv2.addWeighted(img_bgr, 1 - alpha, heat_bgr, alpha, 0)


def cam_to_color_overlay(img_bgr, cam_u8, alpha=0.45):
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    ov = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    return heat, ov


def mask_to_boundary(mask01: np.ndarray, k=3):
    """
    mask01: (H,W) {0,1}
    boundary = dilation - erosion
    """
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(mask01.astype(np.uint8), kernel, iterations=1)
    ero = cv2.erode(mask01.astype(np.uint8), kernel, iterations=1)
    bd = (dil - ero) > 0
    return bd.astype(np.uint8)


# ===============================
# CAM (on align activation)
# ===============================
def gradcam_from_align_act(align_act_bchw: torch.Tensor, out_hw):
    """
    align_act_bchw: [B,C,h,w], requires align_act_bchw.grad exists
    returns cam_u8: [H,W] uint8
    """
    assert align_act_bchw is not None, "align activation is None"
    assert align_act_bchw.grad is not None, "align activation grad is None (backward not called?)"

    A = align_act_bchw[0]          # [C,h,w]
    G = align_act_bchw.grad[0]     # [C,h,w]

    w = G.mean(dim=(1, 2))         # [C]
    cam = torch.relu((w[:, None, None] * A).sum(dim=0))  # [h,w]

    cam -= cam.min()
    cam /= (cam.max() + 1e-6)
    cam = (cam.detach().cpu().numpy() * 255).astype(np.uint8)
    cam = cv2.resize(cam, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return cam


# ===============================
# Pred mask -> RGB
# ===============================
def get_palette(model, num_classes: int):
    """
    mmseg 0.24: model.PALETTE or model.cfg / dataset config
    """
    palette = None
    if hasattr(model, "PALETTE") and model.PALETTE is not None:
        palette = model.PALETTE

    # 也有人把 palette 写在 cfg 里（不保证存在）
    if palette is None:
        try:
            # 可能在 cfg.data.train / test 中；这里做尽量尝试
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
    # mmseg0.x: cfg.data.test.pipeline
    test_pipeline = cfg.data.test.pipeline
    return Compose(test_pipeline)


def _unwrap(x):
    """兼容 DataContainer / list / tensor"""
    if hasattr(x, "data"):  # DataContainer
        return x.data
    return x


def forward_logits_with_grad(model, img_path, device):
    """
    Return:
      logits: [1, C, H, W] (Tensor on device)
      pred_idx: (H,W) uint8
      img_bgr: (H,W,3) uint8
      H,W: original image size (for saving)
    """
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

    # forward (关键：不要 torch.no_grad)
    model.eval()
    logits = model.encode_decode(img_tensor, img_metas)  # [1,C,H,W]

    pred_idx = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    H, W = img_bgr.shape[:2]

    return logits, pred_idx, img_bgr, H, W


# ===============================
# main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='runs/potsdam/vers/SFTran-2/SFTran.large.512x512.potsdam.160k.py')
    parser.add_argument('--checkpoint', default='runs/potsdam/vers/SFTran-2/best_mIoU_iter_57000.pth')
    parser.add_argument('--img', default='/home/dell/x/data/potsdam/img_dir/val/4_14_5488_2560_6000_3072.png')
    parser.add_argument('--outdir', default='work_dirs/feature_maps_mfc_msc')
    parser.add_argument('--device', default='cuda:0')

    # CAM params
    parser.add_argument('--cls-id', type=int, default=1, help='target class id for CAM (e.g., Building)')
    parser.add_argument('--bd-k', type=int, default=3, help='boundary kernel size for B-CAM')
    parser.add_argument('--alpha', type=float, default=0.45, help='overlay alpha')
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

    # register hooks
    bb_hook = FeatureHook()
    h_bb = model.backbone.register_forward_hook(bb_hook)

    fce_hook = FeatureHook()
    h_fce = model.decode_head.FCE.register_forward_hook(fce_hook)

    fuse_hook = FeatureHook()
    h_fuse = model.decode_head.fuse.register_forward_hook(fuse_hook)

    ham_hook = FeatureHook()
    h_ham = model.decode_head.hamburger.register_forward_hook(ham_hook)

    # align: need grad for CAM
    align_act = ActHook(need_grad=True)
    h_align = model.decode_head.align.register_forward_hook(align_act)

    # forward (with grad)
    logits, pred_idx, img_bgr, H, W = forward_logits_with_grad(model, args.img, args.device)

    # save pred mask idx + rgb
    cv2.imwrite(os.path.join(args.outdir, "pred_mask_idx.png"), pred_idx)

    num_classes = logits.shape[1]
    palette = get_palette(model, num_classes)
    pred_rgb = idx_to_rgb(pred_idx, palette)          # RGB
    pred_bgr = pred_rgb[..., ::-1].copy()             # BGR for cv2
    cv2.imwrite(os.path.join(args.outdir, "pred_mask_rgb.png"), pred_bgr)

    # dump backbone stage0-3
    feats = bb_hook.feats
    if feats is None or (not isinstance(feats, (list, tuple))) or len(feats) < 4:
        raise RuntimeError(f"❌ backbone 输出异常: {type(feats)}, len={len(feats) if isinstance(feats,(list,tuple)) else 'NA'}")

    for i in range(4):
        f = feats[i]
        if f.dim() == 4:
            f = f[0]  # [C,h,w]
        np.save(os.path.join(args.outdir, f"stage{i}.npy"), f.detach().cpu().numpy())

        heat = feature_to_heatmap(f, out_hw=(H, W))
        cv2.imwrite(os.path.join(args.outdir, f"stage{i}.png"), heat)

        ov = overlay(img_bgr, heat, alpha=args.alpha)
        cv2.imwrite(os.path.join(args.outdir, f"stage{i}_overlay.png"), ov)

        print(f"✅ stage{i} saved: {tuple(f.shape)}")

    # dump FCE (list of 4)
    fce_feats = fce_hook.feats
    if not isinstance(fce_feats, (list, tuple)):
        raise RuntimeError(f"❌ FCE 输出不是 list/tuple: {type(fce_feats)}")

    for i, f in enumerate(fce_feats):
        if f.dim() == 4:
            f = f[0]  # [C,h,w]
        np.save(os.path.join(args.outdir, f"FCE_stage{i}.npy"), f.detach().cpu().numpy())

        heat = feature_to_heatmap(f, out_hw=(H, W))
        cv2.imwrite(os.path.join(args.outdir, f"FCE_stage{i}.png"), heat)

        ov = overlay(img_bgr, heat, alpha=args.alpha)
        cv2.imwrite(os.path.join(args.outdir, f"FCE_stage{i}_overlay.png"), ov)

    # dump fuse / hamburger / align feature maps
    def save_feat(feat_bchw: torch.Tensor, name: str):
        if feat_bchw is None:
            print(f"[WARN] {name} is None")
            return
        feat = feat_bchw[0] if feat_bchw.dim() == 4 else feat_bchw
        np.save(os.path.join(args.outdir, f"{name}.npy"), feat.detach().cpu().numpy())

        heat = feature_to_heatmap(feat, out_hw=(H, W))
        cv2.imwrite(os.path.join(args.outdir, f"{name}.png"), heat)

        ov = overlay(img_bgr, heat, alpha=args.alpha)
        cv2.imwrite(os.path.join(args.outdir, f"{name}_overlay.png"), ov)

    save_feat(fuse_hook.feats, "HLFT_fuse")
    save_feat(ham_hook.feats, "HLFT_hamburger")
    save_feat(align_act.act, "HLFT_align")

    # ===============================
    # D-CAM / B-CAM on ALIGN
    # ===============================
    cls_id = args.cls_id
    if cls_id < 0 or cls_id >= num_classes:
        raise ValueError(f"--cls-id={cls_id} out of range [0, {num_classes-1}]")

    cls_mask = (pred_idx == cls_id).astype(np.uint8)
    if cls_mask.sum() < 10:
        print(f"[WARN] cls_id={cls_id} 区域像素太少，跳过 CAM。换 patch 或换类。")
    else:
        # ---- D-CAM：类区域目标 ----
        model.zero_grad(set_to_none=True)
        if align_act.act is not None and align_act.act.grad is not None:
            align_act.act.grad.zero_()

        cls_bool = torch.from_numpy(cls_mask).to(logits.device).bool()
        score_d = logits[0, cls_id][cls_bool].mean()
        score_d.backward(retain_graph=True)

        cam_d = gradcam_from_align_act(align_act.act, out_hw=(H, W))
        heat_d, ov_d = cam_to_color_overlay(img_bgr, cam_d, alpha=args.alpha)
        cv2.imwrite(os.path.join(args.outdir, f"DCAM_align_cls{cls_id}.png"), heat_d)
        cv2.imwrite(os.path.join(args.outdir, f"DCAM_align_cls{cls_id}_overlay.png"), ov_d)

        # ---- B-CAM：边界目标 ----
        bd = mask_to_boundary(cls_mask, k=args.bd_k)
        if bd.sum() < 10:
            print(f"[WARN] 边界像素太少，跳过 B-CAM。可尝试 --bd-k 5")
        else:
            model.zero_grad(set_to_none=True)
            if align_act.act is not None and align_act.act.grad is not None:
                align_act.act.grad.zero_()

            bd_bool = torch.from_numpy(bd).to(logits.device).bool()
            score_b = logits[0, cls_id][bd_bool].mean()
            score_b.backward()

            cam_b = gradcam_from_align_act(align_act.act, out_hw=(H, W))
            heat_b, ov_b = cam_to_color_overlay(img_bgr, cam_b, alpha=args.alpha)
            cv2.imwrite(os.path.join(args.outdir, f"BCAM_align_cls{cls_id}.png"), heat_b)
            cv2.imwrite(os.path.join(args.outdir, f"BCAM_align_cls{cls_id}_overlay.png"), ov_b)

            # 可选保存 mask / boundary
            cv2.imwrite(os.path.join(args.outdir, f"pred_mask_cls{cls_id}.png"), cls_mask * 255)
            cv2.imwrite(os.path.join(args.outdir, f"pred_boundary_cls{cls_id}.png"), bd * 255)

        print("✅ D-CAM / B-CAM saved.")

    # remove hooks
    h_bb.remove()
    h_fce.remove()
    h_fuse.remove()
    h_ham.remove()
    h_align.remove()

    print("🎉 All dumps finished.")


if __name__ == "__main__":
    main()
