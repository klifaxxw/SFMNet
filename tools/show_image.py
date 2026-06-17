# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # ========================
# # 工具函数
# # ========================
# def norm01(x):
#     x = x.astype(np.float32)
#     x = x - x.min()
#     return x / (x.max() + 1e-8)
#
# def soft_disk_mask(h, w, r, feather=10):
#     cy, cx = h // 2, w // 2
#     yy, xx = np.ogrid[:h, :w]
#     dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
#     return 1.0 / (1.0 + np.exp((dist - r) / (feather + 1e-6)))
#
# def soft_ring_mask(h, w, r_in, r_out, feather=10):
#     cy, cx = h // 2, w // 2
#     yy, xx = np.ogrid[:h, :w]
#     dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
#
#     def sstep(x, e, k):
#         return 1.0 / (1.0 + np.exp(-(x - e) / (k + 1e-6)))
#
#     inner = sstep(dist, r_in, feather / 3)
#     outer = sstep(dist, r_out, feather / 3)
#     return np.clip(inner - outer, 0, 1)
#
# def band_emphasis(base, mask,
#                   outside_dim=0.30,
#                   inside_boost=0.60,
#                   inside_contrast=1.7):
#     outside = base * outside_dim
#     mid = 0.5
#     inside = np.clip((base - mid) * inside_contrast + mid + inside_boost, 0, 1)
#     out = outside * (1 - mask) + inside * mask
#     return np.clip(out, 0, 1)
#
# def save_pure(img, path):
#     plt.figure(figsize=(4,4))
#     plt.imshow(img, cmap='gray', vmin=0, vmax=1)
#     plt.axis('off')
#     plt.gca().set_position([0,0,1,1])
#     plt.savefig(path, dpi=300, bbox_inches=None, pad_inches=0)
#     plt.close()
#
# # ========================
# # 输出目录
# # ========================
# out_dir = "freq_bands"
# os.makedirs(out_dir, exist_ok=True)
#
# # ========================
# # 读取图像 & 频谱
# # ========================
# img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
# assert img is not None, "没读到 test.png"
#
# h, w = img.shape
# F = np.fft.fftshift(np.fft.fft2(img))
# base = norm01(np.log(np.abs(F) + 1.0))
# base = np.power(base, 0.7)  # 提亮暗部纹理
#
# # ========================
# # 频带设置（圆环已细一半）
# # ========================
# bands = [
#     ("low_freq_circle",  "disk", (0, 35)),        # 低频：实心圆
#     ("mid_freq_ring",    "ring", (55, 75)),       # 中频：细圆环
#     ("high_freq_ring",   "ring", (105, 135)),     # 高频：细圆环
# ]
#
# # ========================
# # 生成并保存
# # ========================
# for name, mode, r in bands:
#     if mode == "disk":
#         mask = soft_disk_mask(h, w, r=r[1], feather=12)
#     else:
#         mask = soft_ring_mask(h, w, r_in=r[0], r_out=r[1], feather=10)
#
#     out = band_emphasis(
#         base, mask,
#         outside_dim=0.30,
#         inside_boost=0.60,
#         inside_contrast=1.7
#     )
#
#     save_pure(out, os.path.join(out_dir, f"{name}.png"))
#
# print(f"✔ 已保存到文件夹：{out_dir}/")



import os
import cv2
import numpy as np
import torch
import torch.fft as fft


# ----------------------------
# Utils
# ----------------------------
def to_tensor_img(img_bgr: np.ndarray, device="cpu"):
    """BGR uint8 -> torch float32 [0,1], shape: (1,3,H,W) with RGB order"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).float() / 255.0  # HWC
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # 1CHW
    return x


def save_tensor_as_image(x: torch.Tensor, path: str, mode="gray", robust=True):
    """
    x: (1,1,H,W) or (1,3,H,W), float
    mode: 'gray' or 'rgb'
    robust: robust normalize (percentile) to make mid/high freq more readable
    """
    x = x.detach().float().cpu().squeeze(0)  # (C,H,W) or (H,W)

    def robust_norm(t: torch.Tensor, p=99.5):
        t = t.float()
        lo = torch.quantile(t.flatten(), (100 - p) / 100.0)
        hi = torch.quantile(t.flatten(), p / 100.0)
        if (hi - lo) < 1e-8:
            return torch.zeros_like(t)
        t = torch.clamp(t, lo, hi)
        return (t - lo) / (hi - lo)

    def minmax_norm(t: torch.Tensor):
        t = t.float()
        mn, mx = t.min(), t.max()
        if (mx - mn) < 1e-8:
            return torch.zeros_like(t)
        return (t - mn) / (mx - mn)

    norm_fn = robust_norm if robust else minmax_norm

    if mode == "gray":
        if x.ndim == 3:  # (3,H,W) -> gray
            xg = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
        else:
            xg = x
        xg = norm_fn(xg)
        img = (xg.numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(path, img)

    elif mode == "rgb":
        if x.ndim == 2:
            x = x.unsqueeze(0).repeat(3, 1, 1)
        xr = norm_fn(x)
        img = (xr.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # HWC RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
    else:
        raise ValueError("mode must be 'gray' or 'rgb'")


def save_spectrum(x: torch.Tensor, path: str, to_gray=True):
    """
    保存 log 幅度频谱图 (fftshift + log1p(abs))
    x: (1,C,H,W)
    """
    if x.shape[1] == 3 and to_gray:
        x = (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).unsqueeze(1)

    F = fft.fft2(x, dim=(-2, -1))
    F = fft.fftshift(F, dim=(-2, -1))
    mag = torch.log1p(torch.abs(F))

    mag = mag.squeeze()
    mag = mag - mag.min()
    mag = mag / (mag.max() + 1e-6)

    img = (mag.detach().cpu().numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(path, img)


# ----------------------------
# Frequency components
# ----------------------------
def sobel_high_freq_fft(x: torch.Tensor):
    """
    复用你 SFourierHighTrans 的频域 Sobel 思路：输出“边缘梯度幅值”高频图
    x: (B,C,H,W)
    return: (B,C,H,W)
    """
    device, dtype = x.device, x.dtype
    b, c, h, w = x.shape

    sobelh = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32, device=device)
    sobelv = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32, device=device)

    fx = torch.zeros((h, w), dtype=dtype, device=device)
    fy = torch.zeros((h, w), dtype=dtype, device=device)
    fx[:3, :3] = sobelh.to(dtype)
    fy[:3, :3] = sobelv.to(dtype)

    FX = fft.fftshift(fft.fft2(fx)).view(1, 1, h, w)
    FY = fft.fftshift(fft.fft2(fy)).view(1, 1, h, w)

    G = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    Gx, Gy = G * FX, G * FY

    gx = torch.abs(fft.ifft2(fft.ifftshift(Gx, dim=(-2, -1)), dim=(-2, -1)))
    gy = torch.abs(fft.ifft2(fft.ifftshift(Gy, dim=(-2, -1)), dim=(-2, -1)))
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    return g


def gaussian_low_freq_fft(x: torch.Tensor, sigma_ratio=0.06):
    """
    频域高斯低通：输出低频分量
    sigma_ratio: sigma / min(H,W)
    """
    device, dtype = x.device, x.dtype
    b, c, h, w = x.shape

    cy, cx = h // 2, w // 2
    yy = torch.arange(h, device=device).view(h, 1).repeat(1, w)
    xx = torch.arange(w, device=device).view(1, w).repeat(h, 1)
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    sigma = sigma_ratio * min(h, w)
    sigma2 = (sigma ** 2) + 1e-12
    mask = torch.exp(-dist2 / (2.0 * sigma2)).to(dtype).view(1, 1, h, w)

    G = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    Glow = G * mask
    low = fft.ifft2(fft.ifftshift(Glow, dim=(-2, -1)), dim=(-2, -1)).real
    return low


def dog_mid_freq_fft(x: torch.Tensor, sigma1_ratio=0.03, sigma2_ratio=0.10):
    """
    DoG 中频：mid = low(sigma2) - low(sigma1), sigma2 > sigma1
    一般：
      sigma1 小 -> 保留更多细节
      sigma2 大 -> 更平滑
    差分后得到“中频/带通”
    """
    assert sigma2_ratio > sigma1_ratio, "sigma2_ratio must be larger than sigma1_ratio"
    low1 = gaussian_low_freq_fft(x, sigma_ratio=sigma1_ratio)
    low2 = gaussian_low_freq_fft(x, sigma_ratio=sigma2_ratio)
    mid = low2 - low1
    return mid, low1, low2


# ----------------------------
# Main
# ----------------------------
def main():
    img_path = "/home/dell/x/data/vaihingen/img_dir/val/area27_512_2048_1024_2560.png"   # 改成你的图
    out_dir = "runs/map/outputs_freq_dog"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    x = to_tensor_img(img_bgr, device=device)  # (1,3,H,W)
    base = os.path.splitext(os.path.basename(img_path))[0]

    # 频谱图（建议灰度）
    save_spectrum(x, os.path.join(out_dir, f"{base}_spectrum.png"), to_gray=True)

    # 低频（用于展示整体结构）
    low = gaussian_low_freq_fft(x, sigma_ratio=0.06)

    # 高频（Sobel：突出边缘/纹理）
    high_sobel = sobel_high_freq_fft(x)

    # 中频（DoG：平滑带通，避免 Gibbs 振铃）
    # 你可以微调 sigma1/sigma2：
    #   sigma1 0.02~0.05, sigma2 0.08~0.15（视分辨率和patch大小）
    mid_dog, low_sigma1, low_sigma2 = dog_mid_freq_fft(x, sigma1_ratio=0.03, sigma2_ratio=0.10)

    # 可选：严格互补高频（低通残差）
    high_residual = x - low

    # 保存
    save_tensor_as_image(x,            os.path.join(out_dir, f"{base}_input.png"), mode="rgb", robust=False)
    save_tensor_as_image(low,          os.path.join(out_dir, f"{base}_low.png"), mode="rgb", robust=False)

    # 中频/高频建议用 robust 归一化，不然太灰
    save_tensor_as_image(mid_dog,      os.path.join(out_dir, f"{base}_mid_dog.png"), mode="rgb", robust=True)
    save_tensor_as_image(high_sobel,   os.path.join(out_dir, f"{base}_high_sobel.png"), mode="rgb", robust=True)

    # 可选保存：两个尺度低通（帮助你解释 DoG）
    save_tensor_as_image(low_sigma1,   os.path.join(out_dir, f"{base}_low_sigma1.png"), mode="rgb", robust=False)
    save_tensor_as_image(low_sigma2,   os.path.join(out_dir, f"{base}_low_sigma2.png"), mode="rgb", robust=False)

    # 可选保存：残差高频
    save_tensor_as_image(high_residual, os.path.join(out_dir, f"{base}_high_residual.png"), mode="rgb", robust=True)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
