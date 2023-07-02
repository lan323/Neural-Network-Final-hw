# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\admin\PycharmProjects\pythonProject3\final_1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from tqdm import tqdm
import argparse
import os
import logging
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image


class CIFAR10Pair(CIFAR10):
    def __init__(self, root, download, train):
        super().__init__(root=root, download=download, train=train)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        img1 = self.train_transform(img)
        img2 = self.train_transform(img)
        return img1, img2

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



def init_logging(path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)

    formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def check_and_mkdir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

class Net(nn.Module):
    def __init__(self, resnet, out_dim):
        super().__init__()
        self.resnet = resnet
        self.g = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )
    def forward(self, x):
        x = self.resnet(x)
        x = self.g(x)
        return F.normalize(x, dim=-1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-out", type=int, help="resnet output dimension", default=256)
    parser.add_argument("-device", type=str, help="device of model and data", default="cuda:0")
    parser.add_argument("-batch_size", type=int, help="batch size", default=10)#128
    parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("-epoch", type=int, help="total epoch for self supervised training", default=3)#500
    parser.add_argument("-t", type=float, help="temperature", default=0.5)
    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)
    model = Net(resnet=ResNet18(), out_dim=args.out)
    log_root = "C:/Users/admin/PycharmProjects/pythonProject3/final_1/self-supervised_log"
    model_pt_root = "C:/Users/admin/PycharmProjects/pythonProject3/final_1/self-supervised_model_pt"
    check_and_mkdir(log_root)
    check_and_mkdir(model_pt_root)

    name = "data_result"
    log_name = name + ".log"
    log_path = os.path.join(log_root, log_name)
    model_pt_path = os.path.join(model_pt_root, name)
    check_and_mkdir(model_pt_path)

    logger = init_logging(log_path)

    dataset = CIFAR10Pair(root="./data/cifar10", download=False, train=True)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    for epoch in range(args.epoch):
        epoch_loss = 0.0
        total_num = 0
        for data in tqdm(dataloader):
            img1, img2 = data
            batch_size = img1.shape[0]
            feat1 = model(img1)
            feat2 = model(img2)
            feat = torch.cat([feat1, feat2], dim=0)
            sim = torch.exp(torch.mm(feat, feat.t().contiguous()) / args.t)
            mask = (torch.ones_like(sim) - torch.eye(2 * batch_size)).bool()
            sim = sim.masked_select(mask).view(2 * batch_size, -1)
            pos_sim = torch.exp(torch.sum(feat1 * feat2, dim=-1) / args.t)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim.sum(dim=-1))).mean()
            epoch_loss += loss.item() * batch_size
            total_num += batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"epoch {epoch}, loss: {epoch_loss / total_num:.3f}")
        resnet_model_saved_path = os.path.join(model_pt_path, f"self-supervised{epoch+1}.pt")
        torch.save(model.resnet.state_dict(), resnet_model_saved_path)
            







    