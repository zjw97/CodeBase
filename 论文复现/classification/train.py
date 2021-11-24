import argparse
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from models import vgg16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-gpu", action='store_true', default=False)
    parser.add_argument("-lr", type=float, default=0.01)
    parser.add_argument("-epoch", type=int, default=64)
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

def train_one_epoch(epoch, model, optimizer, scheduler, criterion, train_dl):
    losses = AveragerMeter()
    batch_time = AveragerMeter()
    accuracy = AveragerMeter()
    for iter, (img, label) in enumerate(train_dl):
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
        batch_time.update(time.time() - start, 1)
        _, pred = out.max(dim=1)
        accuracy.update(torch.sum(pred == label).item() / len(img), len(img))

        if iter % 200 == 0:
            print("Train epoch: %d[%d / %d] loss: %f  acc: %f" %(epoch, iter, len(train_dl), loss.item(), accuracy.avg))


def eval(epoch, model, criterion, val_dl):
    losses = AveragerMeter()
    accuracy = AveragerMeter()
    with torch.no_grad():
        for iter, (img, label) in enumerate(val_dl):
            img = img.cuda()
            label = label.cuda()
            pred = model(img)
            loss = criterion(pred, label)
            losses.update(loss.item())
            accuracy.update(torch.sum(pred == label), len(img))

    print("Eval opoch: %d [%d / %d] loss: %f  acc: %d" % epoch, iter, len(val_dl, loss.item(), accuracy.avg))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu else "cpu")

    model = vgg16(num_classes=10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [32, 56], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    train_dl, val_dl = load_cifar10(args.batch_size)

    for epoch in range(args.epoch):
        model.train()
        train_one_epoch(epoch, model, optimizer, lr_scheduler, criterion, train_dl)
        lr_scheduler.step()
        model.eval()
        eval(epoch, model, criterion, val_dl)

