# tools/show_Agnostic_Backbone_AlignMerged.py
# ------------------------------------------------------------
# mmsegmentation==0.24 compatible
#
# Outputs (ONLY):
# 1) pred_mask_idx.png + pred_mask_rgb.png
# 2) Backbone 4-stage class-agnostic heatmap + overlay
#    - stage{i}_heat.png / stage{i}_heat_overlay.png
#    - AG_backbone_stage{i}_{reduce_mode}.png / _overlay.png (same values, different names)
# 3) ALIGN multi-class merged D-CAM / B-CAM (one image)
#    - DCAM_align_multi_{merge}_cls{tag}.png / _overlay.png
#    - BCAM_align_multi_{merge}_cls{tag}.png / _overlay.png
# ------------------------------------------------------------

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


# =========================
# Basic viz utils
# =========================
def colormap(u8_map: np.ndarray):
    return cv2.applyColorMap(u8_map, cv2.COLORMAP_JET)

def overlay(img_bgr: np.ndarray, heat_bgr: np.ndarray, alpha=0.45):
    heat_bgr = cv2.resize(heat_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
    return cv2.addWeighted(img_bgr, 1 - alpha, heat_bgr, alpha, 0)

def to_u8_with_clip(x_hw: torch.Tensor, out_hw, p_low=0.02, p_high=0.98):
    """
    x_hw: [h,w] float -> uint8 [H,W] with percentile clipping.
    """
    x = x_hw.detach().float()
    vmin = torch.quantile(x, p_low)
    vmax = torch.quantile(x, p_high)
    x = x.clamp(vmin.item(), vmax.item())
    x = (x - vmin) / (vmax - vmin + 1e-6)
    u8 = (x.cpu().numpy() * 255).astype(np.uint8)
    u8 = cv2.resize(u8, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return u8

def save_heat_overlay(outdir: str, prefix: str, img_bgr: np.ndarray, u8_map: np.ndarray,
                      save_heat=True, save_overlay=True, alpha=0.45):
    os.makedirs(outdir, exist_ok=True)
    heat_color = colormap(u8_map)
    if save_heat:
        cv2.imwrite(os.path.join(outdir, f"{prefix}.png"), heat_color)
    if save_overlay:
        cv2.imwrite(os.path.join(outdir, f"{prefix}_overlay.png"),
                    overlay(img_bgr, heat_color, alpha=alpha))


# =========================
# mmseg 0.24 data -> logits with grad
# =========================
def _unwrap(x):
    if hasattr(x, "data"):  # DataContainer
        return x.data
    return x

def forward_logits_with_grad(model, img_path, device):
    """
    Return:
      logits: [1, C, H, W] (Tensor on device)
      pred_idx: (H,W) uint8
      img_bgr: (H,W,3) uint8
      H,W: original image size
    """
    cfg = model.cfg
    pipeline = Compose(cfg.data.test.pipeline)

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


# =========================
# Hooks
# =========================
class BackboneListActHook:
    """store backbone outputs(list/tuple of 4 tensors) and retain grad for each stage if needed"""
    def __init__(self, need_grad=False, expected_len=4):
        self.acts = None
        self.need_grad = need_grad
        self.expected_len = expected_len

    def __call__(self, module, inputs, outputs):
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"Expected list/tuple, got {type(outputs)}")
        if self.expected_len is not None and len(outputs) != self.expected_len:
            raise RuntimeError(f"Expected len={self.expected_len}, got {len(outputs)}")
        self.acts = list(outputs)
        if self.need_grad:
            for t in self.acts:
                if isinstance(t, torch.Tensor):
                    t.retain_grad()

class ActHook:
    """store activation and retain grad (for CAM)"""
    def __init__(self, need_grad=False):
        self.act = None
        self.need_grad = need_grad

    def __call__(self, module, inputs, outputs):
        self.act = outputs
        if self.need_grad and isinstance(outputs, torch.Tensor):
            outputs.retain_grad()


# =========================
# Palette + pred rgb
# =========================
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


# =========================
# Backbone class-agnostic reduction
# =========================
def reduce_feature_to_map(feat_bchw: torch.Tensor, reduce_mode="l2"):
    """
    feat_bchw: [B,C,h,w] -> map_hw: [h,w]
    """
    x = feat_bchw[0]  # [C,h,w]
    if reduce_mode == "mean":
        m = x.mean(dim=0)
    elif reduce_mode == "absmean":
        m = x.abs().mean(dim=0)
    elif reduce_mode == "l2":
        # RMS over channels (稳一些，不容易受 C 大小影响)
        m = torch.sqrt((x * x).mean(dim=0) + 1e-12)
    elif reduce_mode == "max":
        m = x.max(dim=0)[0]
    else:
        raise ValueError(f"Unknown reduce_mode: {reduce_mode}")
    return m

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


def dump_backbone_agnostic_maps(img_bgr, feats_list, outdir,
                               reduce_mode="l2",
                               log=False, demean=False,
                               p_low=0.02, p_high=0.98,
                               alpha=0.45):
    """
    Save two naming sets (same values):
      stage{i}_heat.png / stage{i}_heat_overlay.png
      AG_backbone_stage{i}_{reduce_mode}.png / _overlay.png
    """
    H, W = img_bgr.shape[:2]
    for i in range(4):
        feat = feats_list[i]  # [1,C,h,w]
        resp = reduce_feature_to_map(feat, reduce_mode=reduce_mode)  # [h,w]

        if demean:
            resp = resp - resp.mean()
            resp = resp.abs()

        if log:
            resp = torch.log1p(resp)

        u8 = to_u8_with_clip(resp, out_hw=(H, W), p_low=p_low, p_high=p_high)

        save_heat_overlay(outdir, f"stage{i}_heat", img_bgr, u8,
                          save_heat=True, save_overlay=True, alpha=alpha)
        save_heat_overlay(outdir, f"AG_backbone_stage{i}_{reduce_mode}", img_bgr, u8,
                          save_heat=True, save_overlay=True, alpha=alpha)

    
def dump_backbone_l2_maps(img_bgr, feats_list, outdir, alpha):
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


# =========================
# Grad-CAM on ALIGN + merge multi-class
# =========================
def gradcam_from_act(act_bchw: torch.Tensor, out_hw):
    """
    act_bchw: [B,C,h,w], act_bchw.grad must exist
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

def _merge_cams_u8(cam_list_u8, merge="max"):
    cams = np.stack(cam_list_u8, axis=0).astype(np.float32)  # [K,H,W]
    if merge == "max":
        merged = cams.max(axis=0)
    elif merge == "sum":
        merged = cams.sum(axis=0)
    else:
        raise ValueError("merge must be 'max' or 'sum'")
    merged -= merged.min()
    merged /= (merged.max() + 1e-6)
    return (merged * 255).astype(np.uint8)

def dump_align_dcam_bcam_multi(model, logits, pred_idx, img_bgr, align_act_bchw,
                              outdir, cls_ids, bd_k=3, alpha=0.45,
                              min_pixels=10, merge="max"):
    """
    merged multi-class D-CAM / B-CAM on ALIGN activation
    """
    H, W = img_bgr.shape[:2]
    num_classes = logits.shape[1]

    if align_act_bchw is None:
        print("[WARN] ALIGN activation is None, skip.")
        return

    # ------------- D-CAM -------------
    dcam_list, valid_d = [], []
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

    # ------------- B-CAM -------------
    bcam_list, valid_b = [], []
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

    print("✅ ALIGN multi-class merged D/B-CAM saved.")


# =========================
# Helpers
# =========================
def parse_int_list(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--device', default='cuda:0')

    # backbone agnostic reduce
    parser.add_argument('--reduce_mode', choices=['mean', 'absmean', 'l2', 'max'], default='l2')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--demean', action='store_true')
    parser.add_argument('--p_low', type=float, default=0.02)
    parser.add_argument('--p_high', type=float, default=0.98)

    # align merged D/B-CAM
    parser.add_argument('--cls-ids', type=str, default='1,2,3,4',
                        help='classes merged into one D/B-CAM image, e.g. "1,2,3,4"')
    parser.add_argument('--bd-k', type=int, default=3)
    parser.add_argument('--min-pixels', type=int, default=10)
    parser.add_argument('--merge', choices=['max', 'sum'], default='max')
    parser.add_argument('--alpha', type=float, default=0.45)

    # align layer path
    parser.add_argument('--align-layer', type=str, default='decode_head.align',
                        help='module path for align layer, default: decode_head.align')

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

    # enable grads (ALIGN CAM needs backward)
    for p in model.parameters():
        p.requires_grad_(True)

    # hook backbone (need grad? not necessary for agnostic, but keep False)
    bb_hook = BackboneListActHook(need_grad=False, expected_len=4)
    h_bb = model.backbone.register_forward_hook(bb_hook)

    # hook align (need grad True)
    mod = model
    for name in args.align_layer.split('.'):
        if not hasattr(mod, name):
            raise AttributeError(f"Cannot find align layer: {args.align_layer} (missing {name})")
        mod = getattr(mod, name)
    align_hook = ActHook(need_grad=True)
    h_align = mod.register_forward_hook(align_hook)

    # forward with grad
    logits, pred_idx, img_bgr, H, W = forward_logits_with_grad(model, args.img, args.device)

    # save pred
    cv2.imwrite(os.path.join(args.outdir, "pred_mask_idx.png"), pred_idx)
    num_classes = logits.shape[1]
    palette = get_palette(model, num_classes)
    pred_rgb = idx_to_rgb(pred_idx, palette)      # RGB
    pred_bgr = pred_rgb[..., ::-1].copy()         # BGR for cv2
    cv2.imwrite(os.path.join(args.outdir, "pred_mask_rgb.png"), pred_bgr)

    # backbone outputs
    feats = bb_hook.acts
    if feats is None or (not isinstance(feats, (list, tuple))) or len(feats) < 4:
        raise RuntimeError(f"❌ backbone 输出异常: {type(feats)}, len={len(feats) if isinstance(feats,(list,tuple)) else 'NA'}")

    # dump backbone heatmaps (class-agnostic)
    dump_backbone_agnostic_maps(
        img_bgr=img_bgr,
        feats_list=feats,
        outdir=args.outdir,
        reduce_mode=args.reduce_mode,
        log=args.log,
        demean=args.demean,
        p_low=args.p_low,
        p_high=args.p_high,
        alpha=args.alpha
    )

    dump_backbone_l2_maps(img_bgr, feats, args.outdir, args.alpha)

    # dump align multi merged D/B-CAM
    cls_ids = parse_int_list(args.cls_ids)
    dump_align_dcam_bcam_multi(
        model=model,
        logits=logits,
        pred_idx=pred_idx,
        img_bgr=img_bgr,
        align_act_bchw=align_hook.act,
        outdir=args.outdir,
        cls_ids=cls_ids,
        bd_k=args.bd_k,
        alpha=args.alpha,
        min_pixels=args.min_pixels,
        merge=args.merge
    )

    # remove hooks
    h_bb.remove()
    h_align.remove()

    print("✅ Done. outdir =", args.outdir)


if __name__ == "__main__":
    main()
