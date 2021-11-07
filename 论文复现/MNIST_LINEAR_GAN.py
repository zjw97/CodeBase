import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.utils import save_image



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epoch", type=int, default=100, help="total epochs to train")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.0002)
    parser.add_argument("-channel", type=int, default=1)
    parser.add_argument("-img_size", type=int, default=28)
    parser.add_argument("-latent_dim", type=int, default=100)
    parser.add_argument("-beta1", type=float, default=0.5)
    parser.add_argument("-beta2", type=float, default=0.999)
    parser.add_argument("-result_dir", type=str, default="./result_images/")
    args = parser.parse_args()
    return args

args = parse_args()
C, H, W = args.channel, args.img_size, args.img_size

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(C * H * W, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.view(args.batch_size, C * H * W)
        x = self.model(x)
        return x

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, C * H * W),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(args.batch_size, C, H, W)
        return x



mnist_dl = DataLoader(dataset=MNIST(root="/home/zjw/Datasets", train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5), std=0.5),
                                    ])), shuffle=True, batch_size=args.batch_size, drop_last=True)

generator = Generator()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
discriminator = Discriminator()
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
# 二分类损失
adverisial_loss = nn.BCELoss()

valid = torch.ones(args.batch_size, 1, requires_grad=False).cuda()
fake = torch.zeros(args.batch_size, 1, requires_grad=False).cuda()

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

for epoch in range(args.epoch):
    for i, (imgs, labels) in enumerate(mnist_dl):

        imgs = imgs.cuda()
        labels = labels.cuda()

        # -------------------------------------------
        # Train Discrimiator
        # -------------------------------------------

        optimizer_D.zero_grad()
        noise = torch.randn((args.batch_size, args.latent_dim)).cuda()
        fake_img = generator(noise)
        PRED_FAKE = discriminator(fake_img)
        loss_fake = adverisial_loss(PRED_FAKE, fake)
        PRED_REAL = discriminator(imgs)
        loss_real = adverisial_loss(PRED_REAL, valid)
        loss = loss_fake + loss_real
        # 反向传播
        loss.backward()
        optimizer_D.step()

        # -------------------------------------------
        # Train Generator
        # -------------------------------------------

        optimizer_G.zero_grad()
        noise = torch.randn((args.batch_size, args.latent_dim)).cuda()
        fake_img = generator(noise)
        PRED_FAKE = discriminator(fake_img)
        loss_G = adverisial_loss(PRED_FAKE, valid)

        # 反向传播
        loss_G.backward()
        optimizer_G.step()

        print("[epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss %f]"
              %(epoch, args.epoch, i, len(mnist_dl), loss.item(), loss_G.item()))
        print("PRED_REAL: %f \t PRED_FAKE: %f" %(torch.mean(PRED_REAL.data.cpu()), torch.mean(PRED_FAKE.data.cpu())))

        batches_done = epoch * len(mnist_dl) + i
        if batches_done % 400 == 0:
            save_image(fake_img, args.result_dir + "%d_%d.jpg"%(epoch, batches_done), nrow=8, normalize=True)

