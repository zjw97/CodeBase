import torch
import torch.nn as nn

# 网络结构
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()

        self.features = features
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, use_bn=False):
    layers = []
    in_channels = 3
    for k in cfg:
        if k == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, k, kernel_size=3, stride=1, padding=1, bias=False)
            if use_bn:
                layers += [conv2d, nn.BatchNorm2d(), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = k
    return nn.Sequential(*layers)

def vgg16(num_classes):
    return VGG(make_layers(cfg["D"]), num_classes=num_classes)