import torch
from torch import cuda
import torch.nn as nn
from time import time
from copy import deepcopy

torch.manual_seed(1)
torch.cuda.manual_seed(1)

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

# 核心函数，把conv和bn的参数写入到fused_conv中，卷积的基本配置不变，包括groups和dilation等
def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma

    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
    )
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

# 整个网络的融合
def fuse_module(m):
    children = list(m.named_children())
    conv = None
    conv_name = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d) and conv:
            bc = fuse(conv, child)
            m._modules[conv_name] = bc
            m._modules[name] = DummyModule()
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            # 递归
            fuse_module(child)


def validate(net, cuda=torch.cuda.is_available()):
    net.eval()
    fused_net = deepcopy(net)
    fused_net.eval()
    fuse_module(fused_net)

    error = 0
    origin_time = 0
    fused_time = 0

    if cuda:
        net.cuda()
        fused_net.cuda()
    n = 1
    with torch.no_grad():
        for _ in range(n):
            x = torch.randn(size=(32, 3, 224, 224))
            if cuda:
                x = x.cuda()

            torch.cuda.synchronize()
            start = time()
            out_origin = net(x)
            torch.cuda.synchronize()
            end = time()
            origin_time += end - start

            torch.cuda.synchronize()
            start = time()
            out_fused = fused_net(x)
            torch.cuda.synchronize()
            end = time()
            fused_time += end - start

            error += (out_origin - out_fused).abs().max().item()
    print(f"origin time: {origin_time / n}s fused time: {fused_time / n}s error:{error / n}")

if __name__ == '__main__':
    import torchvision
    net = torchvision.models.mobilenet_v2(True)
    net.eval()
    validate(net, cuda=False)