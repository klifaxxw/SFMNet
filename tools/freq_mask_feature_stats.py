# tools/freq_mask_feature_stats.py
# mmsegmentation==0.24 compatible
import warnings
warnings.filterwarnings("ignore")
import os
import csv
import argparse
import numpy as np
import cv2
import torch
from mmseg.apis import init_segmentor, inference_segmentor


def mkdir(p): os.makedirs(p, exist_ok=True)

def imread_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def l2_response(feat_chw: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(feat_chw), axis=0) + 1e-12)

def resize_hw(x_hw: np.ndarray, H: int, W: int) -> np.ndarray:
    return cv2.resize(x_hw, (W, H), interpolation=cv2.INTER_LINEAR)

def robust_minmax(x, lo=2.0, hi=98.0):
    vmin = float(np.percentile(x, lo))
    vmax = float(np.percentile(x, hi))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax

def norm01(x, vmin, vmax):
    return (x - vmin) / (vmax - vmin + 1e-12)

def to_uint8(x01):
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)

def apply_jet(x01_hw):
    return cv2.applyColorMap(to_uint8(x01_hw), cv2.COLORMAP_JET)

def overlay(img_bgr, heat_bgr, alpha=0.45):
    out = (1 - alpha) * img_bgr.astype(np.float32) + alpha * heat_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def overlay_mask(img_bgr, mask01_hw, color_bgr, alpha=0.35):
    m = np.clip(mask01_hw, 0, 1)[..., None].astype(np.float32)
    col = np.zeros_like(img_bgr, dtype=np.float32)
    col[:] = np.array(color_bgr, np.float32)[None, None, :]
    out = img_bgr.astype(np.float32) * (1 - alpha * m) + col * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------
# Frequency mask from image
# ----------------------------
def make_freq_masks(img_bgr, mode="sobel", high_percentile=85.0, blur_ksize=21, blur_sigma=0.0):
    """
    Return: mask_high (0/1), mask_low (0/1), and score map for debug.
    - sobel: edges as high-freq proxy
    - gaussian: high = |I - blur(I)|, low = blur energy
    """
    g = to_gray(img_bgr).astype(np.float32) / 255.0
    H, W = g.shape

    if mode == "sobel":
        # Sobel magnitude
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        score = mag
    elif mode == "gaussian":
        # high = residual
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(g, (k, k), blur_sigma)
        score = np.abs(g - blur)
    else:
        raise ValueError("mode must be sobel or gaussian")

    thr = float(np.percentile(score.reshape(-1), high_percentile))
    mask_high = (score >= thr).astype(np.uint8)

    # low mask: take complement then clean (or choose bottom percentile)
    # Here: low = bottom percentile symmetric
    thr_low = float(np.percentile(score.reshape(-1), 100.0 - high_percentile))
    mask_low = (score <= thr_low).astype(np.uint8)

    # optional morphology to remove speckles
    kernel = np.ones((3, 3), np.uint8)
    mask_high = cv2.morphologyEx(mask_high, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_low  = cv2.morphologyEx(mask_low,  cv2.MORPH_OPEN, kernel, iterations=1)

    return mask_high, mask_low, score


# ----------------------------
# Hook backbone outputs
# ----------------------------
class BackboneHook:
    def __init__(self):
        self.outputs = None
    def __call__(self, module, inputs, outputs):
        self.outputs = outputs

@torch.no_grad()
def get_backbone_feats(cfg, ckpt, img_path, device="cuda:0"):
    model = init_segmentor(cfg, ckpt, device=device)
    model.eval()
    hook = BackboneHook()
    h = model.backbone.register_forward_hook(hook)
    _ = inference_segmentor(model, img_path)
    h.remove()

    outs = hook.outputs
    if outs is None:
        raise RuntimeError("Backbone hook got None.")
    if torch.is_tensor(outs):
        outs = [outs]
    outs = list(outs)

    feats = []
    for t in outs:
        feats.append(t.squeeze(0).detach().cpu().float().numpy())  # (C,H,W)
    return feats


def stat_in_mask(x_hw: np.ndarray, mask01_hw: np.ndarray, topk=0.0):
    """
    topk: 0 => mean/median over all masked
          0<topk<=1 => mean of top-k fraction within masked (strong-response focus)
    """
    idx = mask01_hw.astype(bool)
    vals = x_hw[idx]
    if vals.size == 0:
        return dict(mean=np.nan, median=np.nan, topk_mean=np.nan, count=0)
    mean = float(vals.mean())
    median = float(np.median(vals))
    if topk and 0 < topk <= 1.0:
        k = max(1, int(vals.size * topk))
        top = np.partition(vals, -k)[-k:]
        topk_mean = float(top.mean())
    else:
        topk_mean = float("nan")
    return dict(mean=mean, median=median, topk_mean=topk_mean, count=int(vals.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stages", default="0,1,2,3")

    ap.add_argument("--msc_cfg", required=True)
    ap.add_argument("--msc_ckpt", required=True)
    ap.add_argument("--mfc_cfg", required=True)
    ap.add_argument("--mfc_ckpt", required=True)
    ap.add_argument("--mscmfc_cfg", required=True)
    ap.add_argument("--mscmfc_ckpt", required=True)

    ap.add_argument("--freq_mode", default="sobel", choices=["sobel", "gaussian"])
    ap.add_argument("--high_percentile", type=float, default=85.0, help="top percentile as high-freq area")
    ap.add_argument("--topk", type=float, default=0.0, help="0 => all masked; 0.1 => top 10% within masked")

    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--robust_lo", type=float, default=2.0)
    ap.add_argument("--robust_hi", type=float, default=98.0)
    args = ap.parse_args()

    stages = [int(x) for x in args.stages.split(",")]
    mkdir(args.outdir)

    img_bgr = imread_bgr(args.img)
    H, W = img_bgr.shape[:2]

    # 1) build high/low masks from image
    mask_high, mask_low, score = make_freq_masks(
        img_bgr, mode=args.freq_mode, high_percentile=args.high_percentile
    )
    cv2.imwrite(os.path.join(args.outdir, "mask_high.png"), (mask_high * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.outdir, "mask_low.png"), (mask_low * 255).astype(np.uint8))

    ov = img_bgr.copy()
    ov = overlay_mask(ov, mask_high.astype(np.float32), color_bgr=(0, 0, 255), alpha=0.35)  # red=high
    ov = overlay_mask(ov, mask_low.astype(np.float32),  color_bgr=(255, 0, 0), alpha=0.25)  # blue=low
    cv2.imwrite(os.path.join(args.outdir, "mask_overlay.png"), ov)

    # 2) get backbone features for each model
    feats = {}
    feats["MSC"] = get_backbone_feats(args.msc_cfg, args.msc_ckpt, args.img, device=args.device)
    feats["MFC"] = get_backbone_feats(args.mfc_cfg, args.mfc_ckpt, args.img, device=args.device)
    feats["MSC+MFC"] = get_backbone_feats(args.mscmfc_cfg, args.mscmfc_ckpt, args.img, device=args.device)

    # 3) compute responses (L2) and save heatmaps with shared normalization per stage
    resp = {k: {} for k in feats.keys()}
    for s in stages:
        for name in feats.keys():
            r = resize_hw(l2_response(feats[name][s]), H, W)
            resp[name][s] = r

        # shared normalization across 3 models at this stage
        stacked = np.stack([resp["MSC"][s], resp["MFC"][s], resp["MSC+MFC"][s]], axis=0)
        vmin, vmax = robust_minmax(stacked.reshape(-1), lo=args.robust_lo, hi=args.robust_hi)

        for name in feats.keys():
            outp = os.path.join(args.outdir, name.replace("+", "_plus_"))
            mkdir(outp)
            x01 = norm01(resp[name][s], vmin, vmax)
            heat = apply_jet(x01)
            cv2.imwrite(os.path.join(outp, f"stage{s}_heat.png"), heat)
            cv2.imwrite(os.path.join(outp, f"stage{s}_heat_overlay.png"), overlay(img_bgr, heat, alpha=args.alpha))

    # 4) stats in high/low for each stage+model + deltas
    csv_path = os.path.join(args.outdir, "freq_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "stage",
            "low_mean", "low_median", "low_topk_mean", "low_count",
            "high_mean", "high_median", "high_topk_mean", "high_count",
            "high_minus_low", "high_div_low"
        ])

        for s in stages:
            row_cache = {}
            for name in ["MSC", "MFC", "MSC+MFC"]:
                low = stat_in_mask(resp[name][s], mask_low, topk=args.topk)
                high = stat_in_mask(resp[name][s], mask_high, topk=args.topk)

                high_minus_low = high["mean"] - low["mean"] if np.isfinite(high["mean"]) and np.isfinite(low["mean"]) else np.nan
                high_div_low = (high["mean"] / (low["mean"] + 1e-12)) if np.isfinite(high["mean"]) and np.isfinite(low["mean"]) else np.nan

                writer.writerow([
                    name, s,
                    low["mean"], low["median"], low["topk_mean"], low["count"],
                    high["mean"], high["median"], high["topk_mean"], high["count"],
                    high_minus_low, high_div_low
                ])
                row_cache[name] = dict(low=low, high=high, ratio=high_div_low, diff=high_minus_low)

            # Δ stats (MSC+MFC - MSC) and (MSC+MFC - MFC) in terms of ratios
            # we append as extra “models” for convenience
            d1_ratio = row_cache["MSC+MFC"]["ratio"] - row_cache["MSC"]["ratio"]
            d2_ratio = row_cache["MSC+MFC"]["ratio"] - row_cache["MFC"]["ratio"]
            d1_diff  = row_cache["MSC+MFC"]["diff"]  - row_cache["MSC"]["diff"]
            d2_diff  = row_cache["MSC+MFC"]["diff"]  - row_cache["MFC"]["diff"]

            writer.writerow([ "DELTA1_ratio(MSC+MFC-MSC)", s, "", "", "", "", "", "", "", "", d1_diff, d1_ratio ])
            writer.writerow([ "DELTA2_ratio(MSC+MFC-MFC)", s, "", "", "", "", "", "", "", "", d2_diff, d2_ratio ])

    cv2.imwrite(os.path.join(args.outdir, "image.png"), img_bgr)
    print("[OK] Saved masks + per-model heatmaps + freq_stats.csv to:", args.outdir)


if __name__ == "__main__":
    main()
