import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["Transformer_net"]

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, bias=False)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3, stride=1)
        self.in1 = nn.InstanceNorm2d(in_channel)
        self.conv2 = ConvLayer(in_channel, in_channel, 3, stride=1)
        self.in2 = nn.InstanceNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + identity
        return out

class UpSampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, upsample=None):
        super(UpSampleBlock, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // stride
        self.reflection_padding = nn.ReflectionPad2d(reflection_padding)
        self.conv = ConvLayer(in_channel, out_channel, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = F.interpolate(x, mode="nearest", scale_factor=self.upsample)
        out = self.reflection_padding(x_in)
        out = self.conv(out)
        return out

class Transformer_net(nn.Module):

    def __init__(self):
        super(Transformer_net, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=3)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpSampleBlock(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpSampleBlock(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = UpSampleBlock(32, 3, kernel_size=9, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = self.deconv3(x)
        return x