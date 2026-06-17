# Copyright (c) OpenMMLab. All rights reserved.
# mmsegmentation==0.24 compatible
#
# 功能：
# 1) dump backbone 4-stage heatmap + overlay
# 2) dump SIA(after) heatmap + overlay  (auto-detect decode_head.SIA / sia / sia_module; fallback to fuse)
# 3) dump ALIGN GRAM (Grad-CAM) + overlay
#    - 支持多类 GRAM 融合到一张图：--cls-ids "1,2,5" + --merge-mode {max,sum,mean}
# 4) save pred_mask_idx.png + pred_mask_rgb.png (palette auto / fallback random)

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
    out_hw: (H,W)  -> resize to original image
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


def to_cam_u8(cam_hw: torch.Tensor, out_hw):
    cam = cam_hw.detach().float()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    cam = (cam.cpu().numpy() * 255).astype(np.uint8)
    cam = cv2.resize(cam, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return cam


def cam_to_color_overlay(img_bgr, cam_u8, alpha=0.45):
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    ov = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    return heat, ov


# ===============================
# Palette / pred mask -> RGB
# ===============================
def get_palette(model, num_classes: int):
    """
    mmseg0.24: model.PALETTE or model.cfg may contain palette.
    """
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
      logits: [1, C, H, W] (Tensor)
      pred_idx: (H,W) uint8
      img_bgr: (H,W,3) uint8 (original)
      H,W: original image size
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
# GRAM (Grad-CAM) on align activation
# ===============================
def compute_gradcam_from_act_grad(act_bchw: torch.Tensor, out_hw):
    """
    act_bchw: [B,C,h,w], needs act_bchw.grad
    return cam_u8 [H,W]
    """
    assert act_bchw is not None, "align activation is None"
    assert act_bchw.grad is not None, "align activation grad is None (backward not called?)"

    A = act_bchw[0]          # [C,h,w]
    G = act_bchw.grad[0]     # [C,h,w]

    w = G.mean(dim=(1, 2))   # [C]
    cam = torch.relu((w[:, None, None] * A).sum(dim=0))  # [h,w]
    cam_u8 = to_cam_u8(cam, out_hw=out_hw)
    return cam_u8


def backward_score(seg_logits: torch.Tensor, pred_idx_np: np.ndarray, class_id: int, score_mode="mask", retain_graph=True):
    """
    seg_logits: [1,C,H,W]
    pred_idx_np: (H,W) np.uint8
    """
    if class_id < 0 or class_id >= seg_logits.shape[1]:
        raise ValueError(f"class_id={class_id} out of range 0..{seg_logits.shape[1]-1}")

    if score_mode == "mask":
        mask = (pred_idx_np == class_id).astype(np.uint8)
        if mask.sum() < 1:
            # fallback: use whole map
            score = seg_logits[:, class_id].mean()
        else:
            mask_bool = torch.from_numpy(mask).to(seg_logits.device).bool()
            score = seg_logits[0, class_id][mask_bool].mean()
    else:
        score = seg_logits[:, class_id].mean()

    score.backward(retain_graph=retain_graph)



def compute_align_cam_u8_for_class(model, logits, pred_idx, align_act, class_id, score_mode, out_hw, retain_graph=True):
    """
    为某个 class_id 计算 align 的 GRAM(cam_u8)。
    每次都清 grad，保证干净。
    """
    model.zero_grad(set_to_none=True)
    if align_act.act is not None and align_act.act.grad is not None:
        align_act.act.grad.zero_()

    backward_score(logits, pred_idx, class_id, score_mode=score_mode, retain_graph=retain_graph)
    cam_u8 = compute_gradcam_from_act_grad(align_act.act, out_hw=out_hw)
    return cam_u8


# ===============================
# Find SIA module (robust)
# ===============================
def find_sia_module(model):
    """
    Try to locate 'SIA' module in decode_head.
    Return: (module, name_str) or (None, None)
    """
    dh = getattr(model, "decode_head", None)
    if dh is None:
        return None, None

    candidates = ["FCE", "SIA", "sia", "sia_module", "sia_block", "SpectralInformationAggregation"]
    for n in candidates:
        if hasattr(dh, n):
            return getattr(dh, n), f"decode_head.{n}"
    if hasattr(dh, "fuse"):
        return getattr(dh, "fuse"), "decode_head.fuse(fallback)"

    return None, None

def find_fuse_module(model):
    """
    Try to locate 'SIA' module in decode_head.
    Return: (module, name_str) or (None, None)
    """
    dh = getattr(model, "decode_head", None)
    if dh is None:
        return None, None

    candidates = ["fuse", "FUSE"]
    for n in candidates:
        if hasattr(dh, n):
            return getattr(dh, n), f"decode_head.{n}"

    # fallback: sometimes "fuse" is effectively "after SIA" in your head
    if hasattr(dh, "fuse"):
        return getattr(dh, "fuse"), "decode_head.fuse(fallback)"

    return None, None


# ===============================
# main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img', required=True)
    parser.add_argument('--outdir', default='work_dirs/vis_stage_sia_aligngram')
    parser.add_argument('--device', default='cuda:0')

    # visualization params
    parser.add_argument('--alpha', type=float, default=0.45)

    # single-class gram (kept for compatibility)
    parser.add_argument('--cls-id', type=int, default=1)

    # multi-class merged gram
    parser.add_argument('--cls-ids', type=str, default='',
                        help='multi classes for merged GRAM, e.g. "1,2,5". '
                             'If set, will generate merged map (and still saves single --cls-id).')
    
    parser.add_argument('--merge-mode', choices=['max', 'sum', 'mean'], default='max',
                        help='how to merge multiple cams into one: max/sum/mean')

    parser.add_argument('--score-mode', choices=['mask', 'mean'], default='mask',
                        help='mask: use predicted region pixels as target score; mean: use full mean score')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # parse cls-ids
    cls_ids = None
    if args.cls_ids.strip():
        cls_ids = [int(x) for x in args.cls_ids.split(',') if x.strip() != '']
        seen = set()
        cls_ids = [x for x in cls_ids if not (x in seen or seen.add(x))]

    # copy original image
    img_name = os.path.basename(args.img)
    dst_img_path = os.path.join(args.outdir, img_name)
    if not os.path.exists(dst_img_path):
        shutil.copy(args.img, dst_img_path)

    # load model
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    # hooks
    bb_hook = FeatureHook()
    h_bb = model.backbone.register_forward_hook(bb_hook)

    sia_module, sia_name = find_sia_module(model)
    sia_hook = FeatureHook()
    h_sia = None
    if sia_module is not None:
        h_sia = sia_module.register_forward_hook(sia_hook)
        print(f"✅ SIA hook attached on: {sia_name}")
    else:
        print("⚠️ 未找到 SIA / fuse 等模块，SIA后热力图将跳过。")

    fuse_module, fuse_name = find_fuse_module(model)
    fuse_hook = FeatureHook()
    h_fuse = None
    if fuse_module is not None:
        h_fuse = fuse_module.register_forward_hook(fuse_hook)
        print(f"✅ FUSE hook attached on: {fuse_name}")
    else:
        print("⚠️ 未找到 FUSE 模块，FUSE后热力图将跳过。")

    align_act = ActHook(need_grad=True)
    h_align = model.decode_head.align.register_forward_hook(align_act)

    # forward
    torch.set_grad_enabled(True)
    model.zero_grad(set_to_none=True)
    logits, pred_idx, img_bgr, H, W = forward_logits_with_grad(model, args.img, args.device)

    # save pred masks
    cv2.imwrite(os.path.join(args.outdir, "pred_mask_idx.png"), pred_idx)
    num_classes = logits.shape[1]
    palette = get_palette(model, num_classes)
    pred_rgb = idx_to_rgb(pred_idx, palette)          # RGB
    pred_bgr = pred_rgb[..., ::-1].copy()             # BGR
    cv2.imwrite(os.path.join(args.outdir, "pred_mask_rgb.png"), pred_bgr)

    # ---------------- backbone stage0-3 heatmaps ----------------
    feats = bb_hook.feats
    if feats is None or (not isinstance(feats, (list, tuple))) or len(feats) < 4:
        raise RuntimeError(
            f"❌ backbone 输出异常: {type(feats)}, "
            f"len={len(feats) if isinstance(feats,(list,tuple)) else 'NA'}"
        )

    for i in range(4):
        f = feats[i]
        if f.dim() == 4:
            f = f[0]  # [C,h,w]
        # np.save(os.path.join(args.outdir, f"stage{i}.npy"), f.detach().cpu().numpy())

        heat = feature_to_heatmap(f, out_hw=(H, W))
        cv2.imwrite(os.path.join(args.outdir, f"stage{i}_heat.png"), heat)

        ov = overlay(img_bgr, heat, alpha=args.alpha)
        cv2.imwrite(os.path.join(args.outdir, f"stage{i}_heat_overlay.png"), ov)

        print(f"✅ stage{i} heat saved: {tuple(f.shape)}")

    # ---------------- SIA(after) heatmap ----------------
    if sia_hook.feats is not None:
        feats = sia_hook.feats
        print("SIA feats type:", type(feats))

        # 统一整理成 list[Tensor]
        if torch.is_tensor(feats):
            feats_list = [feats]
        elif isinstance(feats, dict):
            feats_list = list(feats.values())
        elif isinstance(feats, (list, tuple)):
            feats_list = list(feats)
        else:
            print(f"⚠️ SIA 输出类型不支持: {type(feats)}，跳过。")
            feats_list = []

        if len(feats_list) == 0:
            print("⚠️ SIA feats_list 为空，跳过。")
        else:
            n = min(4, len(feats_list))  # 只取前4个（你想要4尺度的话）
            for i in range(n):
                f = feats_list[i]
                if f is None:
                    print(f"[WARN] SIA_{i} is None, skip.")
                    continue
                if isinstance(f, (list, tuple, dict)):
                    print(f"[WARN] SIA_{i} is nested type {type(f)}, skip.")
                    continue
                if not torch.is_tensor(f):
                    print(f"[WARN] SIA_{i} is not Tensor: {type(f)}, skip.")
                    continue

                if f.dim() == 4:
                    f = f[0]  # [C,h,w]
                elif f.dim() == 3:
                    pass
                else:
                    print(f"[WARN] SIA_{i} unexpected dim={f.dim()}, shape={tuple(f.shape)}, skip.")
                    continue

                heat = feature_to_heatmap(f, out_hw=(H, W))
                cv2.imwrite(os.path.join(args.outdir, f"SIA_{i}_heat.png"), heat)

                ov = overlay(img_bgr, heat, alpha=args.alpha)
                cv2.imwrite(os.path.join(args.outdir, f"SIA_{i}_heat_overlay.png"), ov)

                print(f"✅ SIA_{i} heat saved: {tuple(f.shape)}")
    else:
        print("⚠️ SIA hook 没捕获到输出（模块未走到/短路/条件分支跳过）。")

    
    if fuse_hook.feats is not None:
        feats = fuse_hook.feats
        print("FUSE feats type:", type(feats))

        # 统一整理成 list[Tensor]
        if torch.is_tensor(feats):
            feats_list = [feats]
        elif isinstance(feats, dict):
            feats_list = list(feats.values())
        elif isinstance(feats, (list, tuple)):
            feats_list = list(feats)
        else:
            print(f"⚠️ FUSE 输出类型不支持: {type(feats)}，跳过。")
            feats_list = []

        if len(feats_list) == 0:
            print("⚠️ FUSE feats_list 为空，跳过。")
        else:
            n = min(4, len(feats_list))  # 只取前4个（你想要4尺度的话）
            for i in range(n):
                f = feats_list[i]
                if f is None:
                    print(f"[WARN] FUSE_{i} is None, skip.")
                    continue
                if isinstance(f, (list, tuple, dict)):
                    print(f"[WARN] FUSE_{i} is nested type {type(f)}, skip.")
                    continue
                if not torch.is_tensor(f):
                    print(f"[WARN] FUSE_{i} is not Tensor: {type(f)}, skip.")
                    continue

                if f.dim() == 4:
                    f = f[0]  # [C,h,w]
                elif f.dim() == 3:
                    pass
                else:
                    print(f"[WARN] FUSE_{i} unexpected dim={f.dim()}, shape={tuple(f.shape)}, skip.")
                    continue

                heat = feature_to_heatmap(f, out_hw=(H, W))
                cv2.imwrite(os.path.join(args.outdir, f"FUSE_{i}_heat.png"), heat)

                ov = overlay(img_bgr, heat, alpha=args.alpha)
                cv2.imwrite(os.path.join(args.outdir, f"FUSE_{i}_heat_overlay.png"), ov)

                print(f"✅ FUSE_{i} heat saved: {tuple(f.shape)}")
    else:
        print("⚠️ FUSE hook 没捕获到输出（模块未走到/短路/条件分支跳过）。")

    # ---------------- ALIGN GRAM (single + merged multi-class) ----------------
    out_hw = (H, W)

    # 单类（保持原行为）
    if args.cls_id < 0 or args.cls_id >= num_classes:
        raise ValueError(f"--cls-id={args.cls_id} out of range [0, {num_classes-1}]")

    cam_u8_single = compute_align_cam_u8_for_class(
        model, logits, pred_idx, align_act,
        class_id=args.cls_id, score_mode=args.score_mode, out_hw=out_hw, retain_graph=True
    )
    heat_s, ov_s = cam_to_color_overlay(img_bgr, cam_u8_single, alpha=args.alpha)
    cv2.imwrite(os.path.join(args.outdir, f"GRAM_align_cls{args.cls_id}.png"), heat_s)
    cv2.imwrite(os.path.join(args.outdir, f"GRAM_align_cls{args.cls_id}_overlay.png"), ov_s)
    # np.save(os.path.join(args.outdir, f"GRAM_align_cls{args.cls_id}.npy"), cam_u8_single)
    print("✅ ALIGN GRAM(single) saved.")

    # 多类融合
    if cls_ids is not None and len(cls_ids) > 0:
        cams = []
        valid_ids = []

        for j, cid in enumerate(cls_ids):
            if cid < 0 or cid >= num_classes:
                print(f"[WARN] cls-id {cid} out of range, skip.")
                continue

            # 可选：mask太小就跳过（避免噪声）
            if args.score_mode == "mask":
                if (pred_idx == cid).sum() < 10:
                    print(f"[WARN] cls-id {cid} mask too small, skip.")
                    continue
            is_last = (j == len(cls_ids) - 1)
            cam_u8 = compute_align_cam_u8_for_class(
                model, logits, pred_idx, align_act,
                class_id=cid, score_mode=args.score_mode, out_hw=out_hw, retain_graph=(not is_last)
            )
            cams.append(cam_u8.astype(np.float32) / 255.0)  # 0~1
            valid_ids.append(cid)

        if len(cams) == 0:
            print("⚠️ No valid classes for merged GRAM.")
        else:
            cams = np.stack(cams, axis=0)  # [K,H,W]

            if args.merge_mode == "max":
                merged = cams.max(axis=0)
            elif args.merge_mode == "mean":
                merged = cams.mean(axis=0)
            else:  # sum
                merged = cams.sum(axis=0)
                merged = merged / (merged.max() + 1e-6)

            merged_u8 = (merged * 255).astype(np.uint8)
            heat_m, ov_m = cam_to_color_overlay(img_bgr, merged_u8, alpha=args.alpha)

            tag = "_".join([str(x) for x in valid_ids])
            cv2.imwrite(os.path.join(args.outdir, f"GRAM_align_merged_{args.merge_mode}_cls{tag}.png"), heat_m)
            cv2.imwrite(os.path.join(args.outdir, f"GRAM_align_merged_{args.merge_mode}_cls{tag}_overlay.png"), ov_m)
            # np.save(os.path.join(args.outdir, f"GRAM_align_merged_{args.merge_mode}_cls{tag}.npy"), merged_u8)

            print(f"✅ ALIGN GRAM(merged:{args.merge_mode}) saved for classes: {valid_ids}")

    # remove hooks
    h_bb.remove()
    if h_sia is not None:
        h_sia.remove()
    if h_fuse is not None:
        h_fuse.remove()
    h_align.remove()

    print(f"🎉 Done. outdir = {args.outdir}")


if __name__ == "__main__":
    main()
