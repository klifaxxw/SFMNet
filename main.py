import torch



if __name__ == '__main__':
    ckpt_path = "weights/mix_ade_pretrained.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(ckpt["state_dict"].keys())