import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchvision.models as models
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 512 if torch.cuda.is_available() else 128

random.seed(2021)
np.random.seed(2021)
torch.random.manual_seed(2021)
torch.cuda.manual_seed(2021)

# loader = transforms.Compose([
#     transforms.Resize((image_size, image_size)),
#     # transforms.ToTensor,
# ])

def image_loader(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img, dtype=torch.float).unsqueeze(dim=0).cuda()
    return img.div_(255)

# 加载图片
style_img = image_loader("/home/zhangming/zjw/CodeBase/论文复现/data/images/neural-style/candy.jpg")
content_img = image_loader("/home/zhangming/zjw/CodeBase/论文复现/data/images/neural-style/lion.jpg")

# 显示图片
def show_img(img, title=None):
    img = img.cpu().squeeze(dim=0).numpy()
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()

# 显示图片
# show_img(style_img, "style image")
# show_img(content_img, "content image")

class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(target):
    B, C, H, W = target.size()

    target = target.view(B * C, H * W)
    G = torch.mm(target, target.t())
    return G.div(B * C * H * W)


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        # 风格特征
        self.target = gram_matrix(target)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval() # eval模式下可以让参数不更新，风格迁移是对输入图片进行修改，而不是训练神经网络
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

content_loss_layers = ["conv_4"]
style_loss_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

def get_sytle_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, style_layers=style_loss_layers,
                               content_layers=content_loss_layers):
    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_%d" % (i)
        if isinstance(layer, nn.ReLU):
            name = "relu_%d" % (i)
            layer = nn.ReLU(inplace=False)
        if isinstance(layer, nn.MaxPool2d):
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
            name = "pool_%d" % (i)
        if isinstance(layer, nn.BatchNorm2d):
            name = "bn_%d" % (i)
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach() #  TODO:试一下去掉这个detach会怎么样
            content_loss = ContentLoss(target)
            model.add_module("contentloss_%d" % (i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()  # 这个保存下来的目标张量不需要更新
            style_loss = StyleLoss(target)
            model.add_module("styleloss_%d"%(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i+1]
    return model, content_losses, style_losses

# input_image = content_img.clone()
input_image = torch.randn((1, 3, 512, 512)).cuda()
show_img(input_image, "input_image") # TODO: 或者可以生成随机噪声来优化

optimizer = torch.optim.LBFGS([input_image])

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img,
                       input_img, num_steps=400, style_weight=5e5, content_weight=10):
    print("Building style transfer model...")
    model, content_losses, style_losses = get_sytle_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    run = [0]
    while run[0] < num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_loss = 0
            content_loss = 0

            for sl in style_losses:
                style_loss += sl.loss

            for cl in content_losses:
                content_loss += cl.loss

            style_loss *= style_weight
            content_loss *= content_weight

            loss = style_loss + content_loss
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("steps: %d" % (run[0]))
                print("content_loss: %f \t style_loss: %f"%(style_loss.item(), content_loss.item()))
                print()
            return style_loss + content_loss

        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img.detach()

output_img = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_image)

show_img(output_img, "result")

















