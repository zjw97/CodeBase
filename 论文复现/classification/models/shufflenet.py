import torch
import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups


    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups

        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

class PointwiseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups, **kwargs):
        super(PointwiseConv2d, self).__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class DepthwiseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups, **kwargs):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class ShufflenetUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride, group_conv=True, groups=3, **kwargs):
        super(ShufflenetUnit, self).__init__()
        self.relu = nn.ReLU(inplace=True)


        self.bottleneck = PointwiseConv2d(in_channels,
                                          int(out_channels / 4),
                                          groups=groups if group_conv else 1,
                                          **kwargs)

        self.channelshuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(int(out_channels / 4),
                                         int(out_channels / 4),
                                         stride=stride,
                                         groups=int(out_channels / 4),
                                         **kwargs)

        self.expand = PointwiseConv2d(int(out_channels / 4),
                                      out_channels,
                                      groups=groups,
                                      **kwargs)

        self.fusion = self._add
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.expand = PointwiseConv2d(int(out_channels / 4),
                                          out_channels - in_channels,
                                          groups=groups,
                                          **kwargs)
            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)


    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffle = self.bottleneck(x)
        shuffle = self.relu(shuffle)
        shuffle = self.channelshuffle(shuffle)
        shuffle = self.depthwise(shuffle)
        shuffle = self.expand(shuffle)

        shuffle = self.fusion(shuffle, shortcut)
        shuffle = self.relu(shuffle)
        return shuffle



class ShuffleNet(nn.Module):

    def __init__(self, num_blocks, groups=3):
        super(ShuffleNet, self).__init__()
        if groups == 1:
            out_channels = [24, 144, 288, 576],
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = nn.Conv2d(3, out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = out_channels[0]


        self.stage2 = self._make_stage(out_channels[1], num_blocks[0], groups, 2)
        self.stage3 = self._make_stage(out_channels[2], num_blocks[1], groups, 3)
        self.stage4 = self._make_stage(out_channels[3], num_blocks[2], groups, 4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[-1], 10)

    def _make_stage(self, out_channels, num_blocks, groups, stage):
        strides = [2] + [1] * (num_blocks -1)
        group_conv = stage > 2

        stage = []

        stage.append(ShufflenetUnit(
            self.in_channels,
            out_channels,
            stride=strides[0],
            groups=groups,
            group_conv=group_conv,
            bias=False
        ))
        self.in_channels = out_channels

        for stride in strides[1:]:
            stage.append(
                ShufflenetUnit(
                self.in_channels,
                out_channels,
                stride,
                groups=groups,
                group_conv=True))
            self.in_channels = out_channels

        return nn.Sequential(*stage)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def shufflenet(groups = 3):
    return ShuffleNet(num_blocks=[4, 8, 4], groups=groups)