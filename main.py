import torch
from mmcv.runner import load_checkpoint, load_state_dict
from mmseg.datasets import build_dataloader, build_dataset


if __name__ == "__main__":

    # path = r"/home/dell/x/model/Mamba-UNet/code/pretrained_ckpt/vmamba_tiny_e292.pth"
    # weight = torch.load(path)
    # model_weight = weight['model']
    #
    # new_state_dict = {}
    # for key, value in model_weight.items():
    #     # 为每个键添加 "backbone." 前缀
    #     new_key = f"backbone.{key}"
    #     new_state_dict[new_key] = value
    # weight['model'] = new_state_dict
    # torch.save(weight, "weights/vmamba_tiny_e292_1.pth")
    #
    # for k, v in new_state_dict.items():
    #     print(k)



    # from mmseg.models import build_segmentor
    # from mmcv import Config
    # from tqdm import tqdm
    # # 加载配置文件
    # config_path = 'local_configs/cswin/cross_swin_tiny_patch4_window7_512x512_160k_soild.py'
    # cfg = Config.fromfile(config_path)
    #
    # # 创建模型
    # model = build_segmentor(
    #     cfg.model,
    #     train_cfg=cfg.get('train_cfg'),
    #     test_cfg=cfg.get('test_cfg')).cuda()
    # random_input = torch.randn(1, 3, 512, 512).cuda()  # Batch=1, Channels=3, Height=512, Width=512
    #
    # model_state_dict = model.state_dict()
    # weight_path = r"weights/cswin.pth"
    # weight_dict = torch.load(weight_path)["state_dict"]
    # load_checkpoint(model, weight_path, strict=False, map_location="cpu")


    # # 验证模型的 forward 方法
    # dataset = build_dataset(cfg.data.train)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=False,
    #     shuffle=False)
    # ## 第二部分
    # efficient_test = False
    # model = model.cuda().eval()
    # data = dataset[0]
    # gt = data['gt_semantic_seg'].data
    # img = data['img'].data[None]
    # img = img.cuda()
    # meta = data['img_metas'].data
    # img_path = meta["filename"]
    # name = meta['ori_filename']
    # newdata = {}
    # metas = [[meta]]
    # imgs = [img]
    # result = model(img=imgs, img_metas=metas, rescale=False,return_loss=False)[0]
    # print(result.shape)

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # 读取彩色图像
    color_image = cv2.imread(r"F:\model_result\feature_maps\SegNext_large\148\grad_cam1\o_148.png")  # BGR 格式
    # color_image1 = cv2.imread(r"F:\model_result\feature_maps\swin-base\148\grad_cam1\grad_cam_model_model_backbone_reshape3.jpg")
    # 转换为灰度图像
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 创建字典存储结果
    results = {}
    # -------------------------------
    # 2. Sobel 边缘检测
    # -------------------------------
    # 灰度图处理
    sobel_x_gray = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_gray = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_gray = cv2.magnitude(sobel_x_gray, sobel_y_gray)
    results['Sobel_Gray'] = cv2.convertScaleAbs(sobel_gray)

    # 对彩色图像直接处理
    sobel_x_rgb = cv2.Sobel(color_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_rgb = cv2.Sobel(color_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_rgb = cv2.magnitude(sobel_x_rgb, sobel_y_rgb)
    results['Sobel_RGB'] = cv2.convertScaleAbs(sobel_rgb)

    # -------------------------------
    # 4. Scharr 滤波器
    # -------------------------------
    # 灰度图处理
    scharr_x_gray = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
    scharr_y_gray = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
    scharr_gray = cv2.magnitude(scharr_x_gray, scharr_y_gray)
    results['Scharr_Gray'] = cv2.convertScaleAbs(scharr_gray)

    # 对彩色图像直接处理
    scharr_x_rgb = cv2.Scharr(color_image, cv2.CV_64F, 1, 0)
    scharr_y_rgb = cv2.Scharr(color_image, cv2.CV_64F, 0, 1)
    scharr_rgb = cv2.magnitude(scharr_x_rgb, scharr_y_rgb)
    results['Scharr_RGB'] = cv2.convertScaleAbs(scharr_rgb)

    # scharr_x_rgb = cv2.Scharr(color_image1, cv2.CV_64F, 1, 0)
    # scharr_y_rgb = cv2.Scharr(color_image1, cv2.CV_64F, 0, 1)
    # scharr_rgb = cv2.magnitude(scharr_x_rgb, scharr_y_rgb)
    # results['Scharr_RGB1'] = cv2.convertScaleAbs(scharr_rgb)

    # -------------------------------
    # 显示结果
    # -------------------------------
    plt.figure(figsize=(12, 12))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 遍历结果字典并显示
    methods = ['Sobel', 'Scharr']
    for i, method in enumerate(methods):
        plt.subplot(2, 3, i * 3 + 2)
        plt.title(f"{method} (Gray)")
        plt.imshow(results[f"{method}_Gray"], cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, i * 3 + 3)
        plt.title(f"{method} (RGB)")
        plt.imshow(cv2.cvtColor(results[f"{method}_RGB"], cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    import cv2
    import matplotlib.pyplot as plt

    # 读取彩色图像
    image = cv2.imread(
        r"F:\model_result\feature_maps\swin-base\148\grad_cam0\o_148.png")

    # 检查是否成功读取图像
    if image is None:
        raise FileNotFoundError("图像未找到，请检查路径是否正确！")

    # -------------------------------
    # 对平滑前影像进行 Scharr 边缘检测
    # -------------------------------

    # 1. 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Scharr 边缘检测 (GRAY)
    scharr_x_gray_pre = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)  # X方向梯度
    scharr_y_gray_pre = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)  # Y方向梯度
    scharr_gray_pre = cv2.magnitude(scharr_x_gray_pre, scharr_y_gray_pre)  # 梯度幅值
    scharr_gray_pre = cv2.convertScaleAbs(scharr_gray_pre)  # 转为8位图像

    # 3. Scharr 边缘检测 (RGB)
    scharr_x_rgb_pre = cv2.Scharr(image, cv2.CV_64F, 1, 0)  # X方向梯度
    scharr_y_rgb_pre = cv2.Scharr(image, cv2.CV_64F, 0, 1)  # Y方向梯度
    scharr_rgb_pre = cv2.magnitude(scharr_x_rgb_pre, scharr_y_rgb_pre)  # 梯度幅值
    scharr_rgb_pre = cv2.convertScaleAbs(scharr_rgb_pre)  # 转为8位图像

    # -------------------------------
    # 高斯平滑处理
    # -------------------------------

    # 高斯滤波参数
    kernel_size = 5  # 核大小
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))  # 椭圆形核

    # -------------------------------
    # 闭操作 (MORPH_CLOSE)
    # -------------------------------
    smoothed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # -------------------------------
    # 对平滑后影像进行 Scharr 边缘检测
    # -------------------------------

    # 1. 转换为灰度图像
    gray_smoothed_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)

    # 2. Scharr 边缘检测 (GRAY)
    scharr_x_gray_post = cv2.Scharr(gray_smoothed_image, cv2.CV_64F, 1, 0)  # X方向梯度
    scharr_y_gray_post = cv2.Scharr(gray_smoothed_image, cv2.CV_64F, 0, 1)  # Y方向梯度
    scharr_gray_post = cv2.magnitude(scharr_x_gray_post, scharr_y_gray_post)  # 梯度幅值
    scharr_gray_post = cv2.convertScaleAbs(scharr_gray_post)  # 转为8位图像

    # 3. Scharr 边缘检测 (RGB)
    scharr_x_rgb_post = cv2.Scharr(smoothed_image, cv2.CV_64F, 1, 0)  # X方向梯度
    scharr_y_rgb_post = cv2.Scharr(smoothed_image, cv2.CV_64F, 0, 1)  # Y方向梯度
    scharr_rgb_post = cv2.magnitude(scharr_x_rgb_post, scharr_y_rgb_post)  # 梯度幅值
    scharr_rgb_post = cv2.convertScaleAbs(scharr_rgb_post)  # 转为8位图像

    # -------------------------------
    # 显示结果
    # -------------------------------

    plt.figure(figsize=(12, 10))

    # 原始图像
    plt.subplot(3, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转为RGB显示
    plt.axis('off')

    # 平滑后的图像
    plt.subplot(3, 2, 2)
    plt.title("Smoothed Image")
    plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB))  # 转为RGB显示
    plt.axis('off')

    # Scharr 边缘检测 (GRAY, 平滑前)
    plt.subplot(3, 2, 3)
    plt.title("Scharr Edges (Gray, Before Smoothing)")
    plt.imshow(scharr_gray_pre, cmap='gray')
    plt.axis('off')

    # Scharr 边缘检测 (GRAY, 平滑后)
    plt.subplot(3, 2, 4)
    plt.title("Scharr Edges (Gray, After Smoothing)")
    plt.imshow(scharr_gray_post, cmap='gray')
    plt.axis('off')

    # Scharr 边缘检测 (RGB, 平滑前)
    plt.subplot(3, 2, 5)
    plt.title("Scharr Edges (RGB, Before Smoothing)")
    plt.imshow(cv2.cvtColor(scharr_rgb_pre, cv2.COLOR_BGR2RGB))  # 转为RGB显示
    plt.axis('off')

    # Scharr 边缘检测 (RGB, 平滑后)
    plt.subplot(3, 2, 6)
    plt.title("Scharr Edges (RGB, After Smoothing)")
    plt.imshow(cv2.cvtColor(scharr_rgb_post, cv2.COLOR_BGR2RGB))  # 转为RGB显示
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------
    # 1. 加载图像
    # -------------------------------
    # 加载 RGB 图像
    image = cv2.imread(
        r"F:\model_result\feature_maps\swin-base\148\grad_cam0\o_148.png",
    )
    image = image[:, :, ::-1]  # 转换为 RGB 格式
    rows, cols, c = image.shape  # 获取图像的形状

    # 对图像执行傅里叶变换
    f_transform = np.fft.fftn(image, axes=(0, 1))  # 2D 傅里叶变换
    f_shift = np.fft.fftshift(f_transform, axes=(0, 1))  # 将低频移到中心

    # -------------------------------
    # 2. 构造 Sobel 和拉普拉斯滤波器
    # -------------------------------
    # Sobel 核（水平和垂直）
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # 水平 Sobel 核
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)  # 垂直 Sobel 核

    # 拉普拉斯核
    laplace_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=np.float32)

    # 扩展 Sobel 和拉普拉斯核到图像大小
    sobel_x_filter = np.zeros((rows, cols, c), dtype=np.float32)
    sobel_y_filter = np.zeros((rows, cols, c), dtype=np.float32)
    laplace_filter = np.zeros((rows, cols, c), dtype=np.float32)

    # 将核放在左上角
    sobel_x_filter[:3, :3, :] = sobel_x[:, :, np.newaxis]
    sobel_y_filter[:3, :3, :] = sobel_y[:, :, np.newaxis]
    laplace_filter[:3, :3, :] = laplace_kernel[:, :, np.newaxis]

    # 对 Sobel 和拉普拉斯核执行傅里叶变换
    sobel_x_fft = np.fft.fftn(sobel_x_filter, axes=(0, 1))
    sobel_x_fft = np.fft.fftshift(sobel_x_fft, axes=(0, 1))

    sobel_y_fft = np.fft.fftn(sobel_y_filter, axes=(0, 1))
    sobel_y_fft = np.fft.fftshift(sobel_y_fft, axes=(0, 1))

    laplace_fft = np.fft.fftn(laplace_filter, axes=(0, 1))
    laplace_fft = np.fft.fftshift(laplace_fft, axes=(0, 1))

    # -------------------------------
    # 3. 应用滤波器
    # -------------------------------
    # Sobel 滤波
    filtered_x = f_shift * sobel_x_fft
    filtered_y = f_shift * sobel_y_fft

    # 拉普拉斯低通滤波
    filtered_low = f_shift * laplace_fft

    # -------------------------------
    # 4. 逆傅里叶变换
    # -------------------------------
    # Sobel 滤波结果
    filtered_x_image = np.abs(np.fft.ifftn(np.fft.ifftshift(filtered_x, axes=(0, 1)), axes=(0, 1)))
    filtered_y_image = np.abs(np.fft.ifftn(np.fft.ifftshift(filtered_y, axes=(0, 1)), axes=(0, 1)))
    filtered_combined = np.sqrt(filtered_x_image ** 2 + filtered_y_image ** 2)

    # 拉普拉斯低通滤波结果
    filtered_low_image = np.abs(np.fft.ifftn(np.fft.ifftshift(filtered_low, axes=(0, 1)), axes=(0, 1)))

    # -------------------------------
    # 5. 可视化结果
    # -------------------------------
    plt.figure(figsize=(15, 8))

    # 原始图像
    plt.subplot(2, 4, 1)
    plt.title("Original Image")
    plt.imshow(image / 255.0)  # 归一化显示
    plt.axis("off")

    # 频谱图
    plt.subplot(2, 4, 2)
    plt.title("Magnitude Spectrum")
    plt.imshow(np.log(np.abs(f_shift) + 1)[:, :, 0], cmap="gray")  # 显示 R 通道频谱
    plt.axis("off")

    # Sobel X 滤波器的频谱
    plt.subplot(2, 4, 3)
    plt.title("Sobel X Filter Spectrum")
    plt.imshow(np.log(np.abs(sobel_x_fft) + 1)[:, :, 0], cmap="gray")
    plt.axis("off")

    # Sobel Y 滤波器的频谱
    plt.subplot(2, 4, 4)
    plt.title("Sobel Y Filter Spectrum")
    plt.imshow(np.log(np.abs(sobel_y_fft) + 1)[:, :, 0], cmap="gray")
    plt.axis("off")

    # 水平方向滤波后的图像
    plt.subplot(2, 4, 5)
    plt.title("Filtered Image (Sobel X)")
    plt.imshow(np.clip(filtered_x_image / np.max(filtered_x_image), 0, 1))
    plt.axis("off")

    # 垂直方向滤波后的图像
    plt.subplot(2, 4, 6)
    plt.title("Filtered Image (Sobel Y)")
    plt.imshow(np.clip(filtered_y_image / np.max(filtered_y_image), 0, 1))
    plt.axis("off")

    # Sobel 合并结果
    plt.subplot(2, 4, 7)
    plt.title("Combined Filtered Image")
    plt.imshow(np.clip(filtered_combined / np.max(filtered_combined), 0, 1))
    plt.axis("off")

    # 拉普拉斯低通滤波后的图像
    plt.subplot(2, 4, 8)
    plt.title("Filtered Image (Laplace Low-pass)")
    plt.imshow(np.clip(filtered_low_image / np.max(filtered_low_image), 0, 1))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------
    # 1. 加载图像
    # -------------------------------
    # 加载灰度图像
    image = cv2.imread(
        r"F:\model_result\feature_maps\swin-base\148\grad_cam1\grad_cam_model_model_decode_head_fpn_bottleneck.jpg",
        cv2.IMREAD_GRAYSCALE)
    rows, cols = image.shape

    # -------------------------------
    # 2. 构造 Sobel 核并扩展到图像大小
    # -------------------------------
    # Sobel 核
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 水平 Sobel 核
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # 垂直 Sobel 核

    # 扩展 Sobel 核到图像大小
    sobel_x_filter = np.zeros((rows, cols), dtype=np.float32)
    sobel_y_filter = np.zeros((rows, cols), dtype=np.float32)

    # 将 Sobel 核放在左上角
    sobel_x_filter[:3, :3] = sobel_x
    sobel_y_filter[:3, :3] = sobel_y

    # -------------------------------
    # 3. 对 Sobel 核执行傅里叶变换
    # -------------------------------
    sobel_x_fft = np.fft.fft2(sobel_x_filter)
    sobel_x_fft = np.fft.fftshift(sobel_x_fft)  # 中心化

    sobel_y_fft = np.fft.fft2(sobel_y_filter)
    sobel_y_fft = np.fft.fftshift(sobel_y_fft)

    # -------------------------------
    # 4. 对图像执行傅里叶变换并滤波
    # -------------------------------
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # 对频率域图像应用 Sobel 滤波器
    filtered_x = f_shift * sobel_x_fft
    filtered_y = f_shift * sobel_y_fft

    # 逆傅里叶变换
    filtered_x_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_x)))
    filtered_y_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_y)))

    # 合并结果
    filtered_combined = np.sqrt(filtered_x_image ** 2 + filtered_y_image ** 2)

    # -------------------------------
    # 5. 可视化结果
    # -------------------------------
    plt.figure(figsize=(15, 8))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # 水平方向滤波结果
    plt.subplot(2, 3, 2)
    plt.title("Filtered (Sobel X)")
    plt.imshow(filtered_x_image, cmap="gray")
    plt.axis("off")

    # 垂直方向滤波结果
    plt.subplot(2, 3, 3)
    plt.title("Filtered (Sobel Y)")
    plt.imshow(filtered_y_image, cmap="gray")
    plt.axis("off")

    # 合并结果
    plt.subplot(2, 3, 4)
    plt.title("Combined Filtered Image")
    plt.imshow(filtered_combined, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    image = cv2.imread(
        r"F:\model_result\feature_maps\umamba\148\grad_cam0\grad_cam_model_model_decode_head_reshape_final.jpg",
    )
    image = image[:, :, ::-1]  # 转换为 RGB 格式
    rows, cols, c = image.shape  # 获取图像的形状

    # -------------------------------
    # 2. 构造 Sobel 和拉普拉斯滤波器
    # -------------------------------
    # Sobel 核（水平和垂直）
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # 水平 Sobel 核
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)  # 垂直 Sobel 核

    # 拉普拉斯核
    laplace_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=np.float32)

    # 扩展 Sobel 和拉普拉斯核到图像大小
    sobel_x_filter = np.zeros((rows, cols), dtype=np.float32)
    sobel_y_filter = np.zeros((rows, cols), dtype=np.float32)
    laplace_filter = np.zeros((rows, cols), dtype=np.float32)

    # 将核放在左上角
    sobel_x_filter[:3, :3] = sobel_x
    sobel_y_filter[:3, :3] = sobel_y
    laplace_filter[:3, :3] = laplace_kernel

    # 对 Sobel 和拉普拉斯核执行傅里叶变换
    sobel_x_fft = np.fft.fft2(sobel_x_filter)
    sobel_x_fft = np.fft.fftshift(sobel_x_fft)

    sobel_y_fft = np.fft.fft2(sobel_y_filter)
    sobel_y_fft = np.fft.fftshift(sobel_y_fft)

    laplace_fft = np.fft.fft2(laplace_filter)
    laplace_fft = np.fft.fftshift(laplace_fft)

    # -------------------------------
    # 3. 对每个通道分别执行滤波
    # -------------------------------
    filtered_x_image = []
    filtered_y_image = []
    filtered_combined_image = []
    filtered_low_image = []

    for i in range(c):  # 分别对 R、G、B 通道进行处理
        # 对通道执行傅里叶变换
        f_transform = np.fft.fft2(image[:, :, i])
        f_shift = np.fft.fftshift(f_transform)

        # Sobel 滤波
        filtered_x = f_shift * sobel_x_fft
        filtered_y = f_shift * sobel_y_fft

        # 拉普拉斯低通滤波
        filtered_low = f_shift * laplace_fft

        # 逆傅里叶变换
        filtered_x_image.append(np.abs(np.fft.ifftn(np.fft.ifftshift(filtered_x))))
        filtered_y_image.append(np.abs(np.fft.ifftn(np.fft.ifftshift(filtered_y))))
        filtered_combined_image.append(
            np.sqrt(filtered_x_image[-1] ** 2 + filtered_y_image[-1] ** 2)
        )
        filtered_low_image.append(np.abs(np.fft.ifftn(np.fft.ifftshift(filtered_low))))

    # 合并通道回彩色图像
    filtered_x_image = np.stack(filtered_x_image, axis=-1)
    filtered_y_image = np.stack(filtered_y_image, axis=-1)
    filtered_combined_image = np.stack(filtered_combined_image, axis=-1)
    filtered_low_image = np.stack(filtered_low_image, axis=-1)

    # -------------------------------
    # 4. 可视化结果
    # -------------------------------
    plt.figure(figsize=(15, 8))

    # 原始图像
    plt.subplot(2, 4, 1)
    plt.title("Original Image")
    plt.imshow(image / 255.0)  # 归一化显示
    plt.axis("off")

    # Sobel X 滤波器的频谱
    plt.subplot(2, 4, 2)
    plt.title("Sobel X Filter Spectrum")
    plt.imshow(np.log(np.abs(sobel_x_fft) + 1), cmap="gray")
    plt.axis("off")

    # Sobel Y 滤波器的频谱
    plt.subplot(2, 4, 3)
    plt.title("Sobel Y Filter Spectrum")
    plt.imshow(np.log(np.abs(sobel_y_fft) + 1), cmap="gray")
    plt.axis("off")

    # 拉普拉斯滤波器的频谱
    plt.subplot(2, 4, 4)
    plt.title("Laplace Filter Spectrum")
    plt.imshow(np.log(np.abs(laplace_fft) + 1), cmap="gray")
    plt.axis("off")

    # 水平方向滤波后的图像
    plt.subplot(2, 4, 5)
    plt.title("Filtered Image (Sobel X)")
    plt.imshow(np.clip(filtered_x_image / np.max(filtered_x_image), 0, 1))
    plt.axis("off")

    # 垂直方向滤波后的图像
    plt.subplot(2, 4, 6)
    plt.title("Filtered Image (Sobel Y)")
    plt.imshow(np.clip(filtered_y_image / np.max(filtered_y_image), 0, 1))
    plt.axis("off")

    # Sobel 合并结果
    plt.subplot(2, 4, 7)
    plt.title("Combined Filtered Image")
    plt.imshow(np.clip(filtered_combined_image / np.max(filtered_combined_image), 0, 1))
    plt.axis("off")

    # 拉普拉斯低通滤波后的图像
    plt.subplot(2, 4, 8)
    plt.title("Filtered Image (Laplace Low-pass)")
    plt.imshow(np.clip(filtered_low_image / np.max(filtered_low_image), 0, 1))
    plt.axis("off")

    plt.tight_layout()
    plt.show()