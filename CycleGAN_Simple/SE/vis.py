import torch
from torch import tensor
import torch.nn as nn

import torchvision

import numpy as np
import matplotlib.image as mpimg

from models import Generator

import PIL

def load_model(path, model_strcuture):
    model_para = torch.load(path)
    model_strcuture.load_state_dict(model_para)
    return model_para, model_strcuture

if __name__ == "__main__":

    path = "./output/20211103/netG_B2A.pth"
    model = Generator(3, 3)
    model_para, model = load_model(path, model)
    layer_name = []
    for param_tensor in model.state_dict():
        layer_name.append(param_tensor)

    print(type(model_para))
    exp = nn.Conv2d(3,64,7)
    exp.weight = nn.Parameter(model_para[layer_name[0]])
    exp.bias = nn.Parameter(model_para[layer_name[1]])

    input = torch.from_numpy(mpimg.imread("./datasets/noise2denoise/train/B/negative0.jpg"))
    input = input.float()
    print(input.size())
    input = input.permute([2,0,1])
    input = torch.unsqueeze(input, 0)
    input = input.cuda()
    input_copy = input
    for i in range(int(len(layer_name) / 2)):
        print("input: ", input_copy.size())
        print(layer_name[2 * i],'\t',model_para[layer_name[2 * i]].size())
        conv_size = list(model_para[layer_name[2 * i]].size())
        conv_Layer = nn.Conv2d(conv_size[1], conv_size[0], conv_size[2])
        conv_Layer.weight = nn.Parameter(model_para[layer_name[2 * i]])
        conv_Layer.bias = nn.Parameter(model_para[layer_name[2 * i + 1]])
        tmp = conv_Layer(input_copy)
        input_copy = tmp
        print("output: ", input_copy.size())
        torchvision.utils.save_image(input_copy.permute([1,0,2,3]),"./output/output"+ str(i) +".png")