import torch



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

    # 加载配置文件
    config_path = 'local_configs/Umamba/unet_mamba_tiny_soild_160k_adwn.py'
    cfg = Config.fromfile(config_path)

    # 创建模型
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # # 打印模型的权重键
    # print("Model state_dict keys:")
    # for key in model.state_dict().keys():
    #     print(key)