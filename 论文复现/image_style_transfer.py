import matplotlib.pyplot as plt
import torch
from torch import nn
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.models.vgg import vgg16, vgg19

img_size = 512 if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)

loader = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def show_image(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze().permute(1, 2, 0).numpy()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

content_image = load_image("/home/zjw/CodeBase/论文复现/data/images/neural-style/lion.jpg")
show_image(content_image, "content_image")
style_image = load_image("/home/zjw/CodeBase/论文复现/data/images/neural-style/candy.jpg")
show_image(style_image, "style_image")

input_image = torch.randn((1, 3, img_size, img_size)).to(device)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    b, c, h, w = input.shape
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div_(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std

content_layers_default = ["conv_4"]
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4"]
def create_transfer_model():
    normalize = Normalization(mean, std)

    cnn = vgg16(pretrained=True).features.to(device)

    model = nn.Sequential(normalize)
    content_losses = []
    style_losses = []

    i = 0
    for m in cnn.children():
        if isinstance(m, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)

        if isinstance(m, nn.ReLU):
            name = "relu_{}".format(i)
            m = nn.ReLU(inplace=False)

        if isinstance(m, nn.MaxPool2d):
            name = "pool_{}".format(i)

        if isinstance(m, nn.BatchNorm2d):
            name = "bn_{}".format(i)

        model.add_module(name, m)

        if name in content_layers_default:
            target = model(content_image)
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers_default:
            target = model(style_image)
            style_loss = StyleLoss(target)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]
    return model, content_losses, style_losses


def run_style_transfer(content_weight=1, style_weight=50000, num_steps=300):
    model, content_losses, style_losses = create_transfer_model()
    optimizer = torch.optim.LBFGS([input_image])

    model.requires_grad_(False)
    input_image.requires_grad_(True)

    print("start optimizing")
    run = [0]
    while run[0] < num_steps:

        def closure():
            with torch.no_grad():
                input_image.clamp_(0, 1)
            model(input_image)

            content_loss = 0
            style_loss = 0

            for sl in style_losses:
                style_loss += sl.loss

            for cl in content_losses:
                content_loss += cl.loss

            style_loss *= style_weight
            content_loss *= content_weight
            loss = style_loss + content_loss

            optimizer.zero_grad()
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"step: {run[0]}, content_loss: {content_loss}, style_loss: {style_loss}")

            return loss

        optimizer.step(closure)

run_style_transfer()
with torch.no_grad():
    input_image.clamp_(0, 1)

show_image(input_image.detach(), "output")
