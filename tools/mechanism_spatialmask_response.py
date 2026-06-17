# -*- coding: utf-8 -*-
"""
Mechanism analysis (NO FFT/DWT):
1) Build high/low masks from INPUT image (Canny/Sobel-based).
2) Hook model.backbone forward outputs to get Stage3/Stage4 (outs[2], outs[3]) for 3 models.
3) Build L2 heatmaps, compute masked response stats, and save comparable figures.

Outputs:
- mask_overlay.png
- mechanism_stage3.png
- mechanism_stage4.png
- stats.csv
"""
import warnings
warnings.filterwarnings("ignore")
import os
import csv
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from mmseg.apis import init_segmentor, inference_segmentor


# ===============================
# Hook helper
# ===============================
class FeatureHook:
    def __init__(self):
        self.feats = None

    def __call__(self, module, inputs, outputs):
        self.feats = outputs


def extract_stage_feats(model, img_path, stage3_idx=2, stage4_idx=3):
    """
    Hook model.backbone forward outputs and return stage3/stage4 features.
    Works well for most 4-stage backbones where backbone returns [s1,s2,s3,s4].
    """
    bb_hook = FeatureHook()
    h_bb = model.backbone.register_forward_hook(bb_hook)

    with torch.no_grad():
        _ = inference_segmentor(model, img_path)

    h_bb.remove()

    outs = bb_hook.feats
    if outs is None:
        raise RuntimeError("Backbone hook captured None. Check if model.backbone is executed.")

    if not isinstance(outs, (list, tuple)):
        raise RuntimeError(
            f"Backbone output type is {type(outs)} (not list/tuple). "
            "Please print its structure and adapt the hook."
        )

    if len(outs) <= max(stage3_idx, stage4_idx):
        raise RuntimeError(
            f"Backbone outputs length={len(outs)}; cannot index "
            f"stage3={stage3_idx}, stage4={stage4_idx}."
        )

    s3 = outs[stage3_idx]
    s4 = outs[stage4_idx]

    # Normalize to [1,C,H,W]
    if s3.dim() == 3:
        s3 = s3.unsqueeze(0)
    if s4.dim() == 3:
        s4 = s4.unsqueeze(0)

    return s3.detach().float().cpu(), s4.detach().float().cpu()


# ===============================
# Masks from INPUT image (no frequency decomposition)
# ===============================
def build_high_low_masks_from_image(
    img_bgr,
    canny1=80,
    canny2=160,
    dilate_ks=5,
    low_percentile=40,
    smooth_before_grad=True
):
    """
    High-frequency mask: Canny edges + dilation (edge band)
    Low-frequency mask : low gradient magnitude pixels (bottom percentile), excluding high mask
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # gradient magnitude (Sobel), optionally after Gaussian smoothing
    gray_for_grad = cv2.GaussianBlur(gray, (5, 5), 1.0) if smooth_before_grad else gray
    gx = cv2.Sobel(gray_for_grad, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_for_grad, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # high-frequency: edge band
    edge = cv2.Canny(gray, canny1, canny2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ks, dilate_ks))
    edge_band = cv2.dilate(edge, k, iterations=1)
    M_high = (edge_band > 0).astype(np.uint8)

    # low-frequency: low-gradient region
    thr_l = np.percentile(grad, low_percentile)  # bottom X%
    M_low = (grad <= thr_l).astype(np.uint8)

    # remove overlaps
    M_low = (M_low & (1 - M_high)).astype(np.uint8)

    return M_high, M_low, grad, edge


def overlay_mask(img_bgr, M, color=(0, 0, 255), alpha=0.35):
    out = img_bgr.copy().astype(np.float32)
    col = np.array(color, np.float32).reshape(1, 1, 3)
    mask = (M > 0)[:, :, None].astype(np.float32)
    out = out * (1 - alpha * mask) + col * (alpha * mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_mask_to(mask_hw_uint8, H, W):
    return cv2.resize(mask_hw_uint8, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


# ===============================
# Heatmap + stats
# ===============================
def l2_heatmap(feat_1chw: torch.Tensor) -> torch.Tensor:
    # [1,C,H,W] -> [H,W]
    return torch.sqrt((feat_1chw ** 2).sum(dim=1)[0] + 1e-12)


def masked_mean(A_hw: torch.Tensor, M_hw_uint8: np.ndarray) -> float:
    m = torch.from_numpy(M_hw_uint8).to(A_hw.device).float()
    denom = float(m.sum().item())
    if denom < 1:
        return float("nan")
    return float((A_hw * m).sum().item() / denom)


# ===============================
# Plotting (unified scale per stage)
# ===============================
def save_grid(fig_path, img_bgr, M_high, M_low, heatmaps_dict, title):
    """
    4 columns:
      0) input + masks (high=red, low=blue)
      1) heatmap (L2)
      2) heatmap × high-mask
      3) heatmap × low-mask
    Rows = models
    """
    rows = list(heatmaps_dict.keys())
    n = len(rows)

    allA = np.stack([heatmaps_dict[r] for r in rows], axis=0)
    vmin, vmax = float(allA.min()), float(allA.max())

    # overlay both masks on input: high=red, low=blue
    ov = overlay_mask(img_bgr, M_high, color=(0, 0, 255), alpha=0.35)   # red (BGR)
    ov = overlay_mask(ov,     M_low,  color=(255, 0, 0), alpha=0.30)    # blue (BGR)

    fig, axes = plt.subplots(n, 4, figsize=(12, 3.2 * n), dpi=200)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, r in enumerate(rows):
        A = heatmaps_dict[r]
        H, W = A.shape

        img_rs = cv2.resize(ov, (W, H), interpolation=cv2.INTER_AREA)
        Mh = resize_mask_to(M_high, H, W)
        Ml = resize_mask_to(M_low,  H, W)

        axes[i, 0].imshow(cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Input + masks (high=red, low=blue)" if i == 0 else "")
        axes[i, 0].set_ylabel(r, fontsize=10)
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(A, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Heatmap (L2 over channels)" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(A * Mh, vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("Heatmap × High-mask" if i == 0 else "")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(A * Ml, vmin=vmin, vmax=vmax)
        axes[i, 3].set_title("Heatmap × Low-mask" if i == 0 else "")
        axes[i, 3].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


# ===============================
# Main
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)

    ap.add_argument("--msc_cfg", required=True)
    ap.add_argument("--msc_ckpt", required=True)

    ap.add_argument("--mfc_cfg", required=True)
    ap.add_argument("--mfc_ckpt", required=True)

    ap.add_argument("--full_cfg", required=True)
    ap.add_argument("--full_ckpt", required=True)

    ap.add_argument("--outdir", default="runs/map/mech_out")
    ap.add_argument("--device", default="cuda:0")

    # mask params
    ap.add_argument("--canny1", type=int, default=80)
    ap.add_argument("--canny2", type=int, default=160)
    ap.add_argument("--dilate_ks", type=int, default=5)
    ap.add_argument("--low_percentile", type=int, default=40)

    # backbone output indices
    ap.add_argument("--stage3_idx", type=int, default=2)
    ap.add_argument("--stage4_idx", type=int, default=3)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    img_bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {args.img}")

    M_high, M_low, grad, edge = build_high_low_masks_from_image(
        img_bgr,
        canny1=args.canny1,
        canny2=args.canny2,
        dilate_ks=args.dilate_ks,
        low_percentile=args.low_percentile,
        smooth_before_grad=True
    )

    # save overlay preview
    ov = overlay_mask(img_bgr, M_high, color=(0, 0, 255), alpha=0.35)
    ov = overlay_mask(ov,     M_low,  color=(255, 0, 0), alpha=0.30)
    cv2.imwrite(os.path.join(args.outdir, "mask_overlay.png"), ov)

    models = [
        ("MSC",     args.msc_cfg,  args.msc_ckpt),
        ("MFC",     args.mfc_cfg,  args.mfc_ckpt),
        ("MSC+MFC", args.full_cfg, args.full_ckpt),
    ]

    heat_s3 = {}
    heat_s4 = {}
    stats_rows = []

    for name, cfg, ckpt in models:
        model = init_segmentor(cfg, ckpt, device=args.device)
        model.eval()

        s3, s4 = extract_stage_feats(
            model,
            args.img,
            stage3_idx=args.stage3_idx,
            stage4_idx=args.stage4_idx
        )

        A3 = l2_heatmap(s3).numpy()
        A4 = l2_heatmap(s4).numpy()
        heat_s3[name] = A3
        heat_s4[name] = A4

        for stage, A_np in [("S3", A3), ("S4", A4)]:
            A = torch.from_numpy(A_np)
            H, W = A.shape
            Mh = resize_mask_to(M_high, H, W)
            Ml = resize_mask_to(M_low,  H, W)
            Rh = masked_mean(A, Mh)
            Rl = masked_mean(A, Ml)
            ratio = Rh / (Rl + 1e-6) if (not np.isnan(Rh) and not np.isnan(Rl)) else float("nan")
            stats_rows.append([name, stage, Rh, Rl, ratio])

    save_grid(
        os.path.join(args.outdir, "mechanism_stage3.png"),
        img_bgr, M_high, M_low, heat_s3,
        title=f"Stage3 response (backbone outs[{args.stage3_idx}])"
    )
    save_grid(
        os.path.join(args.outdir, "mechanism_stage4.png"),
        img_bgr, M_high, M_low, heat_s4,
        title=f"Stage4 response (backbone outs[{args.stage4_idx}])"
    )

    with open(os.path.join(args.outdir, "stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Stage", "R_high", "R_low", "Ratio(Rh/Rl)"])
        w.writerows(stats_rows)

    print("[OK] Saved to:", args.outdir)
    print(" - mask_overlay.png")
    print(" - mechanism_stage3.png")
    print(" - mechanism_stage4.png")
    print(" - stats.csv")


if __name__ == "__main__":
    main()
