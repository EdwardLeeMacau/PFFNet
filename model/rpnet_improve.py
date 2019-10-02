"""
  FileName     [ rpnet_improve.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Dehaze model structure. ]
"""

import torch
import torch.nn as nn
from model.net import ConvLayer, UpsampleConvLayer, ResidualBlock
import torch.nn.functional as F

class ImproveNet(nn.Module):
    def __init__(self, res_blocks=18, activation=nn.LeakyReLU(0.2)):
        super(Net, self).__init__()

        # Convolution Unit
        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)

        # Residual Blocks Unit
        self.dehaze = nn.Sequential()
        for i in range(1, res_blocks + 1):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        # ConvolutionTranspose Unit
        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

        self.relu = activation

    def forward(self, x):
        x      = self.relu(self.conv_input(x))
        res2x  = self.relu(self.conv2x(x))
        res4x  = self.relu(self.conv4x(res2x))
        res8x  = self.relu(self.conv8x(res4x))
        res16x = self.relu(self.conv16x(res8x))

        res_dehaze = res16x
        res16x = self.dehaze(res16x)
        res16x = torch.add(res_dehaze, res16x)

        res16x = self.relu(self.convd16x(res16x))
        res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear', align_corners=True)
        res8x = torch.add(res16x, res8x)

        res8x = self.relu(self.convd8x(res8x))
        res8x = F.interpolate(res8x, res4x.size()[2:], mode='bilinear', align_corners=True)
        res4x = torch.add(res8x, res4x)

        res4x = self.relu(self.convd4x(res4x))
        res4x = F.interpolate(res4x, res2x.size()[2:], mode='bilinear', align_corners=True)
        res2x = torch.add(res4x, res2x)

        res2x = self.relu(self.convd2x(res2x))
        res2x = F.interpolate(res2x, x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.add(res2x, x)

        x = self.conv_output(x)

        return x
