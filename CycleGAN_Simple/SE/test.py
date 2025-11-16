#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image

from models import Generator
from datasets import ImageDataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batchSize', type=int, default=1)
    ap.add_argument('--dataroot', type=str, default='datasets/horse2zebra/')
    ap.add_argument('--input_nc', type=int, default=3)
    ap.add_argument('--output_nc', type=int, default=3)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--n_cpu', type=int, default=8)
    ap.add_argument('--which_direction', '-d', choices=['A2B','B2A','both'], default='A2B')
    ap.add_argument('--results_dir', '-o', type=str, default='results')
    ap.add_argument('--generator_A2B', '--checkpoint_A2B', '--ckpt_A2B',
                    dest='generator_A2B', type=str, default='output/netG_A2B.pth')
    ap.add_argument('--generator_B2A', '--checkpoint_B2A', '--ckpt_B2A',
                    dest='generator_B2A', type=str, default='output/netG_B2A.pth')
    ap.add_argument('--limit', type=int, default=0)
    return ap.parse_args()

def build_loader(opt):
    tfm_list = [
        T.Resize((opt.size, opt.size), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ]
    ds = ImageDataset(opt.dataroot, transforms_=tfm_list, mode='test')
    return DataLoader(ds, batch_size=opt.batchSize, shuffle=False,
                      num_workers=opt.n_cpu, pin_memory=bool(opt.cuda))

def load_gen(chkpt, in_nc, out_nc, device):
    G = Generator(in_nc, out_nc).to(device)
    try:
        sd = torch.load(chkpt, map_location=device, weights_only=True)  # PyTorch ≥2.5
    except TypeError:
        sd = torch.load(chkpt, map_location=device)
    G.load_state_dict(sd, strict=False)
    G.eval()
    return G

def main():
    opt = parse_args()
    device = torch.device('cuda:0' if (opt.cuda and torch.cuda.is_available()) else 'cpu')

    do_A2B = opt.which_direction in ('A2B','both')
    do_B2A = opt.which_direction in ('B2A','both')

    G_A2B = load_gen(opt.generator_A2B, opt.input_nc, opt.output_nc, device) if do_A2B else None
    G_B2A = load_gen(opt.generator_B2A, opt.output_nc, opt.input_nc, device) if do_B2A else None

    out_root = Path(opt.results_dir)
    out_A2B = out_root / 'A2B'
    out_B2A = out_root / 'B2A'
    if do_A2B: out_A2B.mkdir(parents=True, exist_ok=True)
    if do_B2A: out_B2A.mkdir(parents=True, exist_ok=True)

    loader = build_loader(opt)

    total = len(loader)
    with torch.inference_mode():
        for i, batch in enumerate(loader, 1):
            if do_A2B:
                real_A = batch['A'].to(device, non_blocking=True)
                fake_B = (G_A2B(real_A)*0.5 + 0.5).clamp(0,1).cpu()
                for k in range(fake_B.size(0)):
                    save_image(fake_B[k], out_A2B / f"{i:04d}_{k:02d}.png")

            if do_B2A:
                real_B = batch['B'].to(device, non_blocking=True)
                fake_A = (G_B2A(real_B)*0.5 + 0.5).clamp(0,1).cpu()
                for k in range(fake_A.size(0)):
                    save_image(fake_A[k], out_B2A / f"{i:04d}_{k:02d}.png")

            print(f"\rGenerated {i}/{total} batches", end='', flush=True)
            if opt.limit and i >= opt.limit:
                break
    print() 

if __name__ == "__main__":
    main()
