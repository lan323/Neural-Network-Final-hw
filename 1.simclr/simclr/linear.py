# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\admin\PycharmProjects\pythonProject3\final_1")
import torch
import torch.nn as nn
import argparse
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import transforms
from torch.optim import Adam, SGD
from tqdm import tqdm
import torch.nn.functional as F
import logging
import os

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





class Net(nn.Module):
    def __init__(self, resnet, num_class, pretrained_path):
        super().__init__()
        self.resnet = resnet
        self.fc = nn.Linear(512, num_class)
        self.resnet.load_state_dict(torch.load(pretrained_path))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x



def init_logging(path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def check_and_mkdir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, help="batch size", default=128)
    parser.add_argument("-pretrained", type=str, help="path of pretrained resnet")
    parser.add_argument("-device", type=str, help="device of model and data", default="cuda:1")
    parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("-epoch", type=int, help="total epoch of training linear part", default=100)
    args = parser.parse_args()
    
    linear_evaluation_log_root = "./linear_evaluation_log"
    model_pt_root = "./linear_evaluation_model_pt"
    check_and_mkdir(linear_evaluation_log_root)
    check_and_mkdir(model_pt_root)
    temp = args.pretrained.split("/")
    pretrained_file_name = temp[-2] + "_" + temp[-1].strip(".pt")


    log_name = "linear_result"
    logger = init_logging(os.path.join(linear_evaluation_log_root, log_name + ".log"))
    model_pt_path = os.path.join(model_pt_root, log_name)
    check_and_mkdir(model_pt_path)

    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    
    trainset = CIFAR10(root="./data/cifar10", train=True, transform=transform_train, download=False)
    testset = CIFAR10(root="./data/cifar10", train=False, transform=transform_test, download=False)

    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device)
    model = Net(ResNet18(), 10, args.pretrained)
    for param in model.resnet.parameters():
        param.requires_grad = False
        
    optimizer = SGD(model.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    loss_function = nn.CrossEntropyLoss()
    
    best_test_acc = 0.0
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0
        total_number = 0
        for data, target in tqdm(train_dataloader):
            batch_size = data.shape[0]
            data = data
            target = target
            out = model(data)
            loss = loss_function(out, target)
            epoch_loss += loss.item() * batch_size
            total_number += batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"epoch {epoch}, loss:{epoch_loss / total_number:.3f}")

        if ((epoch + 1) % 5 == 0):
            torch.save(model.state_dict(), os.path.join(model_pt_path, f"linear{epoch+1}.pt"))
            model.eval()
            with torch.no_grad():
                total_number = 0
                correct_number = 0
                for data, target in tqdm(test_dataloader):
                    batch_size = data.shape[0]
                    out = model(data)
                    pred = torch.argmax(out, dim=-1)
                    pred = pred.detach().cpu()
                    correct_number += (pred == target).sum()
                    total_number += batch_size
                acc = correct_number / total_number
                logger.info(f"epoch {epoch}, test accuracy: {acc * 100:.2f}%")
                
        scheduler.step()







