import torch

if __name__ == '__main__':
    ckpt_path = "weights/biformer_base_best.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(ckpt["model"].keys())