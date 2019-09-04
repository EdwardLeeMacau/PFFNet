#!/usr/bin/env python
"""
  FileName     [ perceptual.py ]
  PackageName  [ PFFNet ]
  Synopsis     [  ]
"""

import numpy as np
import torch
import torchvision
import torchvision.models as models
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils import model_zoo

__all__ = []
model_urls = {}

import utils
from data import DatasetFromFolder
from model.rpnet import Net


def main():
    vgg16_bn = models.vgg16_bn(pretrained=True)
    resnet   = models.resnet18(pretrained=True)

    print(vgg16_bn.features)
    # print(resnet)

if __name__ == "__main__":
    main()
