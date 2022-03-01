import torch
from thop import profile, clever_format
from torchvision.models import *

model = shufflenet_v2_x1_5().cuda()
input = torch.rand(1, 3, 224, 224).cuda()
flops, params = profile(model, inputs=(input, ), verbose=True)
flops, params = clever_format([flops, params], "%.3f")
print(model.__class__.__name__, flops, params)


model = shufflenet_v2_x2_0().cuda()
input = torch.rand(1, 3, 224, 224).cuda()
flops, params = profile(model, inputs=(input, ), verbose=True)
flops, params = clever_format([flops, params], "%.3f")
print(model.__class__.__name__, flops, params)