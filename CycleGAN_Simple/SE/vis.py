import torch
from torch import tensor
import torch.nn as nn

import torchvision

import numpy as np
import matplotlib.image as mpimg

from pathlib import Path

from models import Generator

import PIL

def load_model(path, model_strcuture, device):
    model_para = torch.load(path, map_location=device)
    model_strcuture.load_state_dict(model_para)
    model_strcuture.eval()
    return model_para, model_strcuture

def register_conv_hooks(model, storage):
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            def make_hook(layer_name):
                def hook(_, __, output):
                    storage.append((layer_name, output.detach().cpu()))
                return hook
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = "./model1/netG_B2A.pth"
    model = Generator(3, 3).to(device)
    model_para, model = load_model(path, model, device)

    print(type(model_para))

    np_img = mpimg.imread("./datasets/noise2denoise/train/B/negative0.jpg").copy()
    input = torch.from_numpy(np_img).float()
    print(input.size())
    input = input.permute([2,0,1])
    input = torch.unsqueeze(input, 0).to(device)

    activations = []
    hooks = register_conv_hooks(model, activations)
    with torch.no_grad():
        _ = model(input)
    for h in hooks:
        h.remove()

    out_dir = Path("./output/vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (name, feat) in enumerate(activations):
        print(f"{name}\t{feat.shape}")
        grid = feat.permute(1,0,2,3)
        torchvision.utils.save_image(grid, out_dir / f"{idx:02d}_{name.replace('.', '_')}.png")
