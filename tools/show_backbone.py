# tools/show_FeatureMap.py
# ------------------------------------------------------------
# Dump Grad-CAM ("GRAM") maps for:
#   - backbone: 4 stages
#   - decode_head.FCE: 4 outputs
#   - decode_head.fuse: 1 output
#   - decode_head.hamburger: 1 output
#   - decode_head.align: 1 output
# Also:
#   - save predicted mask (index) and RGB-colored mask (palette)
#
# Notes:
#   1) Grad-CAM needs gradients -> DO NOT use inference_segmentor (often wrapped with no_grad).
#   2) We capture gradients via tensor.register_hook() on each output tensor (stable for tuple/list outputs).
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
# Viz utils
# =========================
def colormap(cam_u8: np.ndarray):
    return cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

def overlay(img_bgr: np.ndarray, heat_bgr: np.ndarray, alpha=0.45):
    heat_bgr = cv2.resize(heat_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
    return cv2.addWeighted(img_bgr, 1 - alpha, heat_bgr, alpha, 0)

def to_cam_u8(cam_hw: torch.Tensor, out_hw):
    """cam_hw: Tensor [h,w] -> uint8 resized to out_hw (H,W)."""
    cam = cam_hw.detach().float()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    cam = (cam.cpu().numpy() * 255).astype(np.uint8)
    cam = cv2.resize(cam, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return cam

def save_cam_pack(outdir: str, prefix: str, img_bgr: np.ndarray, cam_u8: np.ndarray):
    os.makedirs(outdir, exist_ok=True)
    cam_color = colormap(cam_u8)
    cv2.imwrite(os.path.join(outdir, f"{prefix}.png"), cam_color)
    cv2.imwrite(os.path.join(outdir, f"{prefix}_overlay.png"), overlay(img_bgr, cam_color))
    np.save(os.path.join(outdir, f"{prefix}.npy"), cam_u8)


# =========================
# Palette / mask -> RGB
# =========================
def get_palette_from_model(model):
    """
    Try to get palette from dataset METAINFO (mmseg>=1.x) or model.CLASSES/PALETTE (older).
    Return: palette (list[list[int]] or np.ndarray) or None.
    """
    # 1) mmseg 1.x: dataset METAINFO
    try:
        meta = getattr(model, 'dataset_meta', None)
        if meta and 'palette' in meta:
            return meta['palette']
    except Exception:
        pass

    # 2) some models store PALETTE
    pal = getattr(model, 'PALETTE', None)
    if pal is not None:
        return pal

    # 3) decode_head might store palette
    try:
        pal = getattr(model.decode_head, 'PALETTE', None)
        if pal is not None:
            return pal
    except Exception:
        pass

    return None

def mask_to_rgb(mask_hw: np.ndarray, palette):
    """
    mask_hw: uint8/uint16 [H,W] with class indices
    palette: list of RGB triplets, length >= num_classes
    return: rgb image [H,W,3] uint8
    """
    if palette is None:
        raise ValueError("palette is None. Provide --palette manually or ensure dataset_meta has palette.")
    pal = np.array(palette, dtype=np.uint8)  # [K,3]
    if mask_hw.max() >= len(pal):
        raise ValueError(f"Mask max id {mask_hw.max()} exceeds palette size {len(pal)}.")
    rgb = pal[mask_hw]  # [H,W,3] RGB
    return rgb


# =========================
# Build data (keep grad)
# =========================
def build_test_data(model, img_path, device):
    cfg = model.cfg.copy()
    cfg.data.test.pipeline[0].type = 'LoadImageFromFile'
    pipeline = Compose(cfg.data.test.pipeline)

    data = dict(img_info=dict(filename=img_path), img_prefix=None)
    data = pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    return data

def get_seg_logits(model, data):
    img = data['img'][0]
    img_metas = data['img_metas'][0]
    seg_logits = model.encode_decode(img, img_metas)  # [B,C,H,W]
    return seg_logits


# =========================
# Hooks (stable gradient capture)
# =========================
class ListTensorGradHook:
    """Capture list/tuple outputs and their grads via per-tensor register_hook."""
    def __init__(self, expected_len=None):
        self.expected_len = expected_len
        self.feats = None
        self.grads = None

    def fwd(self, module, inputs, outputs):
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"Expected list/tuple output, got {type(outputs)}")
        if self.expected_len is not None and len(outputs) != self.expected_len:
            raise RuntimeError(f"Expected len={self.expected_len}, got len={len(outputs)}")

        self.feats = list(outputs)
        self.grads = [None] * len(self.feats)

        for i, out in enumerate(self.feats):
            if not torch.is_tensor(out):
                raise TypeError(f"outputs[{i}] is not Tensor: {type(out)}")

            def _make_save(ii):
                def _save_grad(g):
                    self.grads[ii] = g
                return _save_grad

            out.register_hook(_make_save(i))

class TensorGradHook:
    """Capture a single Tensor output and its grad via register_hook."""
    def __init__(self):
        self.feat = None
        self.grad = None

    def fwd(self, module, inputs, outputs):
        out = outputs
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            out = list(out.values())[0]
        if not torch.is_tensor(out):
            raise TypeError(f"Expected Tensor output, got {type(out)}")

        self.feat = out

        def _save_grad(g):
            self.grad = g

        out.register_hook(_save_grad)


# =========================
# Grad-CAM core
# =========================
def compute_cam_from_feat_grad(feat_bchw: torch.Tensor, grad_bchw: torch.Tensor):
    """
    feat_bchw, grad_bchw: [B,C,h,w] -> cam [h,w] normalized.
    """
    w = grad_bchw.mean(dim=(2, 3), keepdim=True)    # [B,C,1,1]
    cam = (w * feat_bchw).sum(dim=1)                # [B,h,w]
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-6)
    return cam[0]                                   # [h,w]

def backward_score(seg_logits: torch.Tensor, class_id: int, score_mode="mask"):
    if class_id < 0 or class_id >= seg_logits.shape[1]:
        raise ValueError(f"class_id={class_id} out of range 0..{seg_logits.shape[1]-1}")

    if score_mode == "mask":
        with torch.no_grad():
            pred = seg_logits.argmax(dim=1)         # [B,H,W]
            mask = (pred == class_id).float()
            if mask.sum() < 1:
                mask = torch.ones_like(mask)
        score = (seg_logits[:, class_id] * mask).sum() / (mask.sum() + 1e-6)
    else:
        score = seg_logits[:, class_id].mean()

    score.backward()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='runs/potsdam/ablation/msc-sia-dgf/SFTran.msc-sia-dgf.512x512.potsdam.160k.py')
    parser.add_argument('--checkpoint', default='runs/potsdam/ablation/msc-sia-dgf/best_mIoU_iter_33000.pth')
    parser.add_argument('--img', default='/home/dell/x/data/potsdam/img_dir/test/4_10_3072_0_3584_512.png')
    parser.add_argument('--outdir', default='work_dirs/msc_grams_all')
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--class_id', type=int, default=1)
    parser.add_argument('--score_mode', choices=['mean', 'mask'], default='mask')

    # dump switches
    parser.add_argument('--dump_backbone_gram', default=True)
    parser.add_argument('--dump_fce_gram', default=True)
    parser.add_argument('--dump_fuse_gram', default=True)
    parser.add_argument('--dump_hamburger_gram', default=True)
    parser.add_argument('--dump_align_gram', default=True)

    # mask RGB options
    parser.add_argument('--dump_pred_mask', action='store_true')
    parser.add_argument('--palette', type=str, default='',
                        help='Optional manual palette, e.g. "0,0,0;255,255,255;0,255,0;..." (RGB triplets). '
                             'If empty, try to load from model.dataset_meta.')

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # copy original image for convenience
    img_name = os.path.basename(args.img)
    dst_img = os.path.join(args.outdir, img_name)
    if not os.path.exists(dst_img):
        shutil.copy(args.img, dst_img)

    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        raise FileNotFoundError(args.img)
    H, W = img_bgr.shape[:2]

    # load model
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()

    # parse manual palette if provided
    manual_palette = None
    if args.palette.strip():
        # format: "r,g,b;r,g,b;..."
        items = args.palette.strip().split(';')
        pal = []
        for it in items:
            rgb = [int(x) for x in it.split(',')]
            assert len(rgb) == 3
            pal.append(rgb)
        manual_palette = pal

    # hooks
    hb = ListTensorGradHook(expected_len=4)
    hfce = ListTensorGradHook(expected_len=4)
    hfuse = TensorGradHook()
    hham = TensorGradHook()
    halign = TensorGradHook()

    handles = []

    if args.dump_backbone_gram:
        handles.append(model.backbone.register_forward_hook(hb.fwd))
    if args.dump_fce_gram:
        handles.append(model.decode_head.FCE.register_forward_hook(hfce.fwd))
    if args.dump_fuse_gram:
        handles.append(model.decode_head.fuse.register_forward_hook(hfuse.fwd))
    if args.dump_hamburger_gram:
        handles.append(model.decode_head.hamburger.register_forward_hook(hham.fwd))
    if args.dump_align_gram:
        handles.append(model.decode_head.align.register_forward_hook(halign.fwd))

    if not (args.dump_backbone_gram or args.dump_fce_gram or args.dump_fuse_gram or
            args.dump_hamburger_gram or args.dump_align_gram or args.dump_pred_mask):
        print("Nothing selected. Use --dump_* flags.")
        return

    # forward with grad enabled
    torch.set_grad_enabled(True)
    model.zero_grad(set_to_none=True)

    data = build_test_data(model, args.img, args.device)
    seg_logits = get_seg_logits(model, data)  # [1,C,H,W]

    # save predicted mask (index + RGB)
    if args.dump_pred_mask:
        pred = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # [H,W]
        cv2.imwrite(os.path.join(args.outdir, "pred_mask_idx.png"), pred)

        # palette: manual > model
        palette = manual_palette if manual_palette is not None else get_palette_from_model(model)
        if palette is not None:
            rgb = mask_to_rgb(pred, palette)          # RGB
            bgr = rgb[..., ::-1]                      # OpenCV wants BGR
            cv2.imwrite(os.path.join(args.outdir, "pred_mask_rgb.png"), bgr)
        else:
            print("⚠️ No palette found. pred_mask_rgb.png not generated. Provide --palette to enable.")

    # backward once to fill all grads
    if any([args.dump_backbone_gram, args.dump_fce_gram, args.dump_fuse_gram,
            args.dump_hamburger_gram, args.dump_align_gram]):
        backward_score(seg_logits, args.class_id, score_mode=args.score_mode)

    # remove module hooks
    for h in handles:
        h.remove()

    # ---------------- backbone 4-stage GRAM ----------------
    if args.dump_backbone_gram:
        if hb.feats is None or hb.grads is None:
            raise RuntimeError("❌ backbone hook did not capture outputs")
        for i in range(4):
            feat = hb.feats[i]
            grad = hb.grads[i]
            if grad is None:
                raise RuntimeError(f"❌ backbone grad[{i}] is None (unexpected).")
            cam_hw = compute_cam_from_feat_grad(feat, grad)
            cam_u8 = to_cam_u8(cam_hw, out_hw=(H, W))
            save_cam_pack(args.outdir, f"GRAM_backbone_stage{i}_cls{args.class_id}", img_bgr, cam_u8)
        print("✅ backbone 4-stage GRAM saved.")

    # ---------------- FCE 4 outputs GRAM ----------------
    if args.dump_fce_gram:
        if hfce.feats is None or hfce.grads is None:
            raise RuntimeError("❌ FCE hook did not capture outputs")
        for i in range(4):
            feat = hfce.feats[i]
            grad = hfce.grads[i]
            if grad is None:
                raise RuntimeError(
                    f"❌ FCE grad[{i}] is None. "
                    f"该输出分支可能未参与最终logits路径或被detach。"
                )
            cam_hw = compute_cam_from_feat_grad(feat, grad)
            cam_u8 = to_cam_u8(cam_hw, out_hw=(H, W))
            save_cam_pack(args.outdir, f"GRAM_FCE_out{i}_cls{args.class_id}", img_bgr, cam_u8)
        print("✅ FCE 4-output GRAM saved.")

    # ---------------- fuse GRAM ----------------
    if args.dump_fuse_gram:
        if hfuse.feat is None or hfuse.grad is None:
            raise RuntimeError("❌ fuse feat/grad not captured (grad None => not used in logits?)")
        cam_hw = compute_cam_from_feat_grad(hfuse.feat, hfuse.grad)
        cam_u8 = to_cam_u8(cam_hw, out_hw=(H, W))
        save_cam_pack(args.outdir, f"GRAM_fuse_cls{args.class_id}", img_bgr, cam_u8)
        print("✅ fuse GRAM saved.")

    # ---------------- hamburger GRAM ----------------
    if args.dump_hamburger_gram:
        if hham.feat is None or hham.grad is None:
            raise RuntimeError("❌ hamburger feat/grad not captured (grad None => not used in logits?)")
        cam_hw = compute_cam_from_feat_grad(hham.feat, hham.grad)
        cam_u8 = to_cam_u8(cam_hw, out_hw=(H, W))
        save_cam_pack(args.outdir, f"GRAM_hamburger_cls{args.class_id}", img_bgr, cam_u8)
        print("✅ hamburger GRAM saved.")

    # ---------------- align GRAM ----------------
    if args.dump_align_gram:
        if halign.feat is None or halign.grad is None:
            raise RuntimeError("❌ align feat/grad not captured (grad None => not used in logits?)")
        cam_hw = compute_cam_from_feat_grad(halign.feat, halign.grad)
        cam_u8 = to_cam_u8(cam_hw, out_hw=(H, W))
        save_cam_pack(args.outdir, f"GRAM_align_cls{args.class_id}", img_bgr, cam_u8)
        print("✅ align GRAM saved.")

    print(f"🎉 Done. outdir = {args.outdir}")


if __name__ == "__main__":
    main()
