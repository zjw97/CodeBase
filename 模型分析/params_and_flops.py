import torch
from thop import profile, clever_format
from torchvision.models import resnet18

model = resnet18().cuda()
input = torch.rand(1, 3, 224, 224).cuda()
flops, params = profile(model, inputs=(input, ), verbose=True)
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)