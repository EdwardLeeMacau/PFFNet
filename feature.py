import argparse
import logging
import logging.config
import os

import numpy as np
import torch
import torchvision
import torchvision.models as models
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

import utils
from data import DatasetFromFolder
from model.rpnet import Net


def main():
    vgg16_bn = models.vgg16_bn(pretrained=True, progress=True)
    resnet   = models.resnet18(pretrained=True, progress=True)

    print(vgg16_bn)
    print(resnet)

if __name__ == "__main__":
    main()
