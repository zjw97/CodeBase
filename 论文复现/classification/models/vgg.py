import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.vgg import vgg16

__all__ = ["vgg16"]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# 网络结构
cfg = {
    'vgg11' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg13' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'vgg19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, arch, num_classes=10, use_bn=False, pretrained=True):
        super(VGG, self).__init__()
        self.arch = arch  # 模型结构
        features = make_layers(cfg[arch])
        self.features = features
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

        if pretrained:
            self.load_pretrained_state_dict()
        else:
            for m in self.modules():
                self.random_init_parameter(m)

        if num_classes != 1000:
            out_layer = nn.Linear(4096, 10)
            self.random_init_parameter(out_layer)
            self.classifier[6] = out_layer


    def load_pretrained_state_dict(self):
        print("************ load pretrained model *******************")
        state_dict = load_state_dict_from_url(model_urls[self.arch])
        self.load_state_dict(state_dict)

    def random_init_parameter(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, "fan_out", nonlinearity="relu")
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

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
            conv2d = nn.Conv2d(in_channels, k, kernel_size=3, stride=1, padding=1)
            if use_bn:
                layers += [conv2d, nn.BatchNorm2d(), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = k
    return nn.Sequential(*layers)

def vgg16(num_classes, pretrained=False, use_bn=False):
    model = VGG("vgg16", num_classes=num_classes, use_bn=use_bn, pretrained=pretrained)
    return model