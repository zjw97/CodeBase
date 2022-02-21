import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter

import utils
from transform_net import Transformer_net
from vgg import Vgg16

content_weight = 1
style_weight = 5e4

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    writer = SummaryWriter("./models")

    np.random.seed(2022)
    torch.random.seed(2022)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = ImageFolder(args.dataset, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=24)
    transformer = Transformer_net().to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    mse_loss = F.mse_loss()

    vgg = Vgg16(requires_grad=False)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])

    style_img = utils.load_image(args.style_image)
    style_img = style_transform(style_img)
    style_img = style_img.repeat((32, 1, 1, 1)).to(device)

    feature_style = vgg(utils.normalize_batch(style_img))
    gram_style = [y for y in feature_style]

    for epoch in range(2):
        pbar = tqdm(total=len(train_loader), postfix=dict, mininterval=0.3)
        count = 0
        for i, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            x_feature = vgg(utils.normalize_batch(x))
            y_feature = vgg(utils.normalize_batch(y))

            content_loss = content_weight * mse_loss(y_feature.relu2_2, x_feature.relu2_2)

            style_loss = 0
            for f_s, f_y in zip(feature_style, y_feature):
                gm_y = utils.gram_matrix(f_y)
                style_loss += mse_loss(gm_y, f_y)

            loss = content_loss + style_weight * style_loss
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{
                "iter": i,
                "content_loss": content_loss,
                "style_loss": style_loss,
                "total_loss": loss
            })

            writer.add_scalar("content loss", content_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar("style loss", style_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar("total loss", loss.item(), epoch * len(train_loader) + 1)



            pbar.update(1)




def stylize(args):
    pass



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommond")

    main_parser = parser.add_subparsers("")
    train_parser = main_parser.add_parser("train")
    train_parser.add_argument("--style-image", default=None, type=str)
    train_parser.add_argument("--dataset", default=None, type=str)
    train_parser.add_argument("--batch-size", default=64, type=int)
    train_parser.add_argument("--cuda", action="store_false", help="weathre use gpu. default: true")

    eval_parser = main_parser.add_parser("eval")
    eval_parser.add_argument("--style_image", default=None, type=str)
    eval_parser.add_argument("--content-image", default=None, type=str)
    eval_parser.add_argument("--model", default=None, type=str)

    args = parser.parse_args("train --style-image --dataset /home/hanglijun/zjw/examples/fast_style_transfer/images/train2017")
    if args.subcommond is None:
        print("Error: Please specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("Error: cuda is not available. please try on cpu")

    if args.subcommod is "train":
        train(args)
    else:
        stylize(args)
    return args

def main():
    args = parse_args()



if __name__ == "__main__":
    main()
