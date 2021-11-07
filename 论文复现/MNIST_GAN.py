import os

import torch.cuda

os.environ["CUDA_VISIABLE_DEVICES"] = "1"
import argparse
from torchvision.utils import save_image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.autograd import Variable
import numpy as np



def arg_parse():
    parser = argparse.ArgumentParser("MNIST_CONV_GAN")
    parser.add_argument("-lr", type=float, default=0.0001, help="learning rate of generative and advisial net")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-beta1", type=float, default=0.9, help="hyperparameter of optimizer Adam")
    parser.add_argument("-beta2", type=float, default=0.999, help="hyperparameter of optimizer Adam")
    parser.add_argument("-channel", type=int, default=1, help="channel of input image")
    parser.add_argument("-img_size", type=int, default=32, help="size of input image")
    parser.add_argument("-latent_dim", type=int, default=256, help="output size of classification network")
    parser.add_argument("-num_epoch", type=int, default=100, help="total epcohs to train")
    parser.add_argument("-sample_interval", type=int, default=400, help="sample train")
    parser.add_argument("-results_dir", type=str, default="./result_images")
    args = parser.parse_args()
    return args


args = arg_parse()

# load dataset
train_ds = MNIST(root="/home/zhangming/Datasets", train=True, download=False,
                 transform=transforms.Compose([
                     transforms.Resize(args.img_size),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5), std=(0.5)),

                 ]))
train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=2, drop_last=True)

C, H, W = args.channel, args.img_size, args.img_size
# generative network
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # self.linear = nn.Linear(100, 4 * 4 * 1024)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=0, bias=False),  # 4 * 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=1, bias=False),  # 8 * 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, input):
        # input = self.linear(input)
        # input = input.view(batch_size, 1024, 4, 4)
        output = self.model(input)
        return output



class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, stride=2, kernel_size=4, padding=1, bias=False), # 64 * 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=4, padding=1, bias=False), # 32 * 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=4, padding=1, bias=False), # 16 * 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.out = nn.Sequential(
            nn.Linear(4 * 4 * 256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        features = self.model(input)
        features = features.view(-1, 4 * 4 * 256),
        features = features[0]
        output = self.out(features)
        return output


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_()



discriminator = Discriminator()
discriminator.apply(init_weight)
generator = Generator()
generator.apply(init_weight)
# loss
discriminator_loss = nn.BCELoss()
generator_loss = nn.BCELoss()

if torch.cuda.is_available():
    discriminator_loss.cuda()
    generator_loss.cuda()
    discriminator.cuda()
    generator.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# train
for epoch in range(args.num_epoch):
    for i, (imgs, _) in enumerate(train_dl):
        batch_size = args.batch_size

        valid = Variable(torch.ones(batch_size, 1).cuda(), requires_grad=False)
        fake = Variable(torch.zeros(batch_size, 1).cuda(), requires_grad=False)

        # real img
        real_img = Variable(imgs.type(torch.FloatTensor).cuda())

        # -------------------------------
        # Train Generator
        # -------------------------------


        optimizer_G.zero_grad()

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, 128, 1, 1))).cuda())
        # generator a batch size of images
        gen_imgs = generator(z)

        PRO_D_fake = discriminator(gen_imgs)
        g_loss = discriminator_loss(PRO_D_fake, valid)  # 训练生成器的时候标签要用valid， 为了加速训练

        g_loss.backward()
        optimizer_G.step()

        # --------------------------
        # Train Discriminator
        # --------------------------

        optimizer_D.zero_grad()

        PRO_D_real = discriminator(real_img)
        PRO_D_fake = discriminator(gen_imgs.detach())

        real_loss = discriminator_loss(PRO_D_real, valid)
        fake_loss = discriminator_loss(PRO_D_fake, fake)

        loss = real_loss + fake_loss
        loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] "
              % (epoch, args.num_epoch, i, len(train_dl), loss.data.cpu(), g_loss.data.cpu()))
        print("[PRO_D_real: %f]\t [PRO_D_fake: %f] " % (torch.mean(PRO_D_real.data.cpu()),
                                                        torch.mean(PRO_D_fake.data.cpu())))

        batches_done = epoch * len(train_dl) + i
        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[: 64], args.results_dir + "/%d-%d.png"
                       % (epoch, batches_done), nrow=8, normalize=True)












