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



    from mmseg.models import build_segmentor
    from mmcv import Config
    from tqdm import tqdm
    # 加载配置文件
    config_path = 'local_configs/cswin/cross_swin_tiny_patch4_window7_512x512_160k_soild.py'
    cfg = Config.fromfile(config_path)

    # 创建模型
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    random_input = torch.randn(1, 3, 512, 512).cuda()  # Batch=1, Channels=3, Height=512, Width=512

    model_state_dict = model.state_dict()
    weight_path = r"weights/cswin.pth"
    weight_dict = torch.load(weight_path)["state_dict"]
    load_checkpoint(model, weight_path, strict=False, map_location="cpu")

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