#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from matplotlib import cm

from models import Generator, SELayer


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize SE-block attention maps")
    ap.add_argument('--checkpoint', type=str, default='model1/netG_B2A.pth')
    ap.add_argument('--image', type=str, default='datasets/noise2denoise/train/B/negative0.jpg')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--output', type=str, default='results/attention')
    ap.add_argument('--cmap', type=str, default='magma', help='Matplotlib colormap for heatmaps')
    return ap.parse_args()


def load_generator(chkpt, device):
    model = Generator(3, 3).to(device)
    state = torch.load(chkpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(img_path, size, device):
    tfm = T.Compose([
        T.Resize((size, size), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = Image.open(img_path).convert('RGB')
    return tfm(img).unsqueeze(0).to(device)


def register_se_hooks(model):
    att_outputs = []
    gate_outputs = []
    handles = []

    def make_att_hook(name):
        def hook(_, __, output):
            att_outputs.append((name, output.detach().cpu()))
        return hook

    def make_gate_hook(name):
        def hook(_, __, output):
            gate_outputs.append((name, output.detach().cpu()))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, SELayer):
            handles.append(module.register_forward_hook(make_att_hook(name)))
            handles.append(module.fc.register_forward_hook(make_gate_hook(name)))
    return handles, att_outputs, gate_outputs


def normalize_tensor(tensor):
    t = tensor.clone()
    t -= t.min()
    denom = t.max().clamp(min=1e-8)
    t /= denom
    return t


def weights_to_grid(gate_vec):
    """
    gate_vec: Tensor of shape (B, C)
    returns: Tensor of shape (B, 1, H, W) forming a square-ish grid
    """
    b, c = gate_vec.shape
    side = math.ceil(math.sqrt(c))
    total = side * side
    grids = []
    for bi in range(b):
        vals = gate_vec[bi]
        if total > c:
            pad = torch.zeros(total - c, dtype=vals.dtype, device=vals.device)
            vals = torch.cat([vals, pad], dim=0)
        grids.append(vals.view(1, side, side))
    return torch.stack(grids, dim=0)


def save_heatmap(tensor, path, cmap_name):
    arr = tensor.squeeze().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    colored = (cmap(arr)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(path)


def main():
    opt = parse_args()
    device = torch.device('cuda:0' if (opt.cuda and torch.cuda.is_available()) else 'cpu')

    model = load_generator(opt.checkpoint, device)
    inp = preprocess(opt.image, opt.size, device)

    handles, att_maps, gate_maps = register_se_hooks(model)
    with torch.no_grad():
        _ = model(inp)
    for h in handles:
        h.remove()

    out_dir = Path(opt.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (name, feat) in enumerate(att_maps):
        spatial = feat.mean(dim=1, keepdim=True)  # average over channels
        spatial = normalize_tensor(spatial)
        base = out_dir / f"{idx:02d}_{name.replace('.', '_')}_spatial"
        save_image(spatial, Path(f"{base}.png"))
        save_heatmap(spatial, Path(f"{base}_heatmap.png"), opt.cmap)

    for idx, (name, gate) in enumerate(gate_maps):
        gate = gate.squeeze(-1).squeeze(-1)  # (B, C)
        gate = normalize_tensor(gate)
        grid = weights_to_grid(gate)
        base = out_dir / f"{idx:02d}_{name.replace('.', '_')}_weights"
        save_image(grid, Path(f"{base}.png"))
        save_heatmap(grid, Path(f"{base}_heatmap.png"), opt.cmap)

    print(f"Saved SE attention maps to {out_dir}")


if __name__ == "__main__":
    main()
