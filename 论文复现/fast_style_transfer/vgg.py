import torch
from torch import nn
from torchvision.models.vgg import vgg16

from collections import namedtuple

__all__ = ["Vgg16"]

class Vgg16(nn.Module):
    def __init__(self, requires_grad):
        super(vgg, self).__init__()
        features = vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for i in range(4):
            self.slice1.add_module(str(i), features[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), features[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), features[i])
        for i in range(9, 23):
            self.slice4.add_module(str(i), features[i])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x

        vgg_output = namedtuple("VGGOutput", ["relu1_2 relu2_2, relu3_3, relu4_3"])
        out = vgg_output(relu1_2, relu2_2, relu3_3, relu4_3)
        return out