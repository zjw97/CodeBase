import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import tensorboardX

from models.vgg import vgg16
from models.resnet import resnet50
from models.shufflenet import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-gpu", action='store_true', default=False)
    parser.add_argument("-lr", type=float, default=0.01)
    parser.add_argument("-weight_decay", type=float, default=1e-4)
    parser.add_argument("-epoch", type=int, default=64)
    parser.add_argument("-save_dir", type=str, default="./vgg_cifar10")
    parser.add_argument("-use_bn", action="store_true", default=False)
    parser.add_argument("-pretrained", action="store_true", default=False)
    parser.add_argument("-random_seed", type=int, default=2021)
    args = parser.parse_args()
    return args


def load_cifar10(batch_size=32):
    # TODO： Normalize 参数没有设置
    train_ds = CIFAR10(root="/home/zjw/Datasets", train=True,
                       transform=transforms.Compose(
                           [transforms.Resize((224, 224)),
                           transforms.ToTensor()]
                       ), download=True)

    val_ds = CIFAR10(root="/home/zjw/Datasets", train=False,
                     transform=transforms.Compose(
                         [transforms.Resize((224, 224)),
                         transforms.ToTensor()]
                     ), download=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return train_dl, val_dl

class AveragerMeter():
    def __init__(self):
        self.val = 0
        self.total = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, n):
        self.val = val
        self.total += n
        self.sum += val * n
        self.avg = self.sum / self.total

def train_one_epoch(epoch, model, optimizer, scheduler, criterion, train_dl, writer):
    losses = AveragerMeter()
    batch_time = AveragerMeter()
    accuracy = AveragerMeter()
    for it, (img, label) in enumerate(train_dl):
        start = time.time()
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        loss = criterion(out, label)



        # 反响传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), len(img))
        writer.add_scalar("train loss", losses.val, epoch * len(train_dl) + it)
        batch_time.update(time.time() - start, 1)
        _, pred = out.max(dim=1)
        accuracy.update(torch.sum(pred == label).item() / len(img), len(img))

        if it % 200 == 0:
            print("Train epoch: %d[%d / %d] learning_rate: %f  loss: %f  acc: %f" %(epoch, it, len(train_dl), optimizer.state_dict()["param_groups"][0]['lr'], loss.item(), accuracy.avg))


def eval(epoch, model, criterion, val_dl, writer):
    losses = AveragerMeter()
    accuracy = AveragerMeter()
    with torch.no_grad():
        for it, (img, label) in enumerate(val_dl):
            img = img.cuda()
            label = label.cuda()
            out = model(img)
            loss = criterion(out, label)
            _, pred = out.max(dim=1)
            losses.update(loss.item(), len(img))
            accuracy.update(torch.sum(pred == label) / len(img), len(img))

        writer.add_scalar("eval accuracy: ", epoch, accuracy.avg)

    print("Eval epoch: %d [%d / %d] loss: %f  acc: %f" %(epoch, it, len(val_dl), loss.item(), accuracy.avg))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu else "cpu")
    writer = tensorboardX.SummaryWriter(logdir=args.save_dir)
    set_random_seed(args.random_seed)
    # model = vgg16(num_classes=10, pretrained=True, use_bn=False).cuda()
    # model = resnet50(num_classes=10).cuda()
    model = shufflenet().cuda()
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [32, 56], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    train_dl, val_dl = load_cifar10(args.batch_size)

    for epoch in range(args.epoch):
        model.train()
        train_one_epoch(epoch, model, optimizer, lr_scheduler, criterion, train_dl, writer)
        lr_scheduler.step()
        model.eval()
        eval(epoch, model, criterion, val_dl, writer)

