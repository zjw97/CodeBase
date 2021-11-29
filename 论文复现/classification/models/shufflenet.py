import torch
import torch.nn as nn

__all__ = ["shufflenet"]


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernerl_size, stride, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernerl_size, stride, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelShuffle(nn.Module):

    def __init__(self, gropus):
        super(ChannelShuffle, self).__init__()
        self.groups = gropus

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
            nn.Conv2d(in_channels, out_channels, 1, groups=groups, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class DepthwiseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, **kwargs):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, **kwargs),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.depthwise(x)


class ShuffleNetUnit(nn.Module):

    def __init__(self, in_channel, out_channels, groups, stride, has_group=True, **kwargs):
        super(ShuffleNetUnit, self).__init__()

        first_module_groups = groups if has_group else 1

        self.bottleneck = PointwiseConv2d(in_channel,
                                          int(out_channels / 4),
                                          first_module_groups,
                                          **kwargs)


        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(int(out_channels / 4),
                                         int(out_channels / 4),
                                         groups=int(out_channels / 4),
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         **kwargs)

        self.expand = PointwiseConv2d(int(out_channels / 4),
                                      out_channels,
                                      groups,
                                      **kwargs)

        self.relu = nn.ReLU(True)

        self.fusion = self._add
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.fusion = self._cat


    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)


    def forward(self, x):

        shuffle = self.bottleneck(x)
        shuffle = self.channel_shuffle(shuffle)
        shuffle = self.depthwise(shuffle)
        shuffle = self.expand(shuffle)

        shortcut = self.shortcut(x)

        fusion = self.fusion(shuffle, shortcut)
        return self.relu(fusion)



class ShuffleNet(nn.Module):

    def __init__(self, num_blocks, groups):
        super(ShuffleNet, self).__init__()
        if groups == 1:
            out_channels = [24, 144, 288, 576]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]
        # 输入feature map 大小224
        self.conv1 = nn.Conv2d(3, out_channels[0], kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = out_channels[0]

        self.stage2 = self._make_stage(out_channels[0], 2, num_blocks[0], groups)
        self.stage3 = self._make_stage(out_channels[1], 3, num_blocks[1], groups)
        self.stage4 = self._make_stage(out_channels[2], 4, num_blocks[2], groups)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[-1], 10)

        for m in self.modules():
            self.random_init_parameter(m)


    def _make_stage(self, out_channels, stage_idx, num_blocks, groups):

        strides = [2] + [1] * (num_blocks - 1)

        stage = []

        has_group = stage_idx > 2
        # stage2 的第一个pointwise不采用group卷积，因为通道数太少
        stage.append(
            ShuffleNetUnit(
                self.in_channels,
                out_channels,
                groups,
                stride=strides[0],
                has_group=has_group,
            )
        )
        self.in_channels = out_channels

        for stride in strides[1:]:
            stage.append(
                ShuffleNetUnit(
                    self.in_channels,
                    out_channels,
                    groups,
                    stride,
                    has_group=True,
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*stage)


    def random_init_parameter(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, "fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def shufflenet(groups=3):
    return ShuffleNet([4, 8, 4], groups)



