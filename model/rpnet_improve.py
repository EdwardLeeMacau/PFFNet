"""
  FileName     [ rpnet_improve.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Dehaze model structure. ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.net import (
    MeanShift, InverseMeanShift, ConvLayer, UpsampleConvLayer, ResidualBlock, UpsampleResidualBlock
)

# TODO: 
# 1. Try to change torch.add() to torch.cat()
#    Remember to change the in_channels of UpsampleConvLayer()
#    - 1x1 conv 
#    - Hard Tuning the Channel Size 

# TODO:
# 2. Change Encoder and Decoder: 
#    - ConvLayer()     -> ResidualBlock()
#       With DownSample ShortCut
#    - ResidualBlock() -> UpsampleResidualBlock()
#       With UpSample ShortCut
class ImproveNet(nn.Module):
    def __init__(self, res_blocks=18, activation=nn.LeakyReLU(0.2)):
        super(ImproveNet, self).__init__()

        self.relu = activation

        # Convolution Unit
        self.conv_input = ConvLayer(  3,  16, kernel_size=11, stride=1, norm_layer=nn.BatchNorm2d)
        self.conv2x     = ConvLayer( 16,  32, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        self.conv4x     = ConvLayer( 32,  64, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        self.conv8x     = ConvLayer( 64, 128, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        self.conv16x    = ConvLayer(128, 256, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)

        # Residual Blocks Unit
        self.dehaze = nn.Sequential()
        for i in range(1, res_blocks + 1):
            self.dehaze.add_module(
                'res%d' % i, 
                ResidualBlock(
                    in_channels=256, 
                    out_channels=256, 
                    kernel_size=3,
                    stride=1
                )
            )

        # ConvolutionTranspose Unit
        
        # Method: Basic Method
        # self.convd16x = UpsampleConvLayer(256, 128, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        # self.convd8x  = UpsampleConvLayer(128,  64, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        # self.convd4x  = UpsampleConvLayer( 64,  32, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        # self.convd2x  = UpsampleConvLayer( 32,  16, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)

        # Method: Hard Tuning
        # self.convd16x = UpsampleConvLayer(512, 128, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        # self.convd8x  = UpsampleConvLayer(256,  64, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        # self.convd4x  = UpsampleConvLayer(128,  32, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)
        # self.convd2x  = UpsampleConvLayer( 64,  16, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d)

        # Method: 1x1 conv
        self.convd16x = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d),
            UpsampleConvLayer(256, 128, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d),
        )
        self.convd8x = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d),
            UpsampleConvLayer(128, 64, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d),
        )
        self.convd16x = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d),
            UpsampleConvLayer(64, 32, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d),
        )
        self.convd16x = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d),
            UpsampleConvLayer(32, 16, kernel_size=5, stride=2, norm_layer=nn.BatchNorm2d),
        )

        self.conv_output = ConvLayer(16, 3, kernel_size=5, stride=1)

    def forward(self, x):
        # Encoder
        x      = self.relu(self.conv_input(x))
        res2x  = self.relu(self.conv2x(x))
        res4x  = self.relu(self.conv4x(res2x))
        res8x  = self.relu(self.conv8x(res4x))
        res16x = self.relu(self.conv16x(res8x))

        # Residual Blocks
        res_dehaze = res16x
        res16x = self.dehaze(res16x)
        res16x = torch.add(res_dehaze, res16x)
        # res16x = torch.cat((res_dehaze, res16x), 1)

        # Decoder
        res16x = self.relu(self.convd16x(res16x))
        res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear', align_corners=True)
        # res8x  = torch.add(res16x, res8x)
        res8x = torch.cat((res16x, res8x), 1)

        res8x  = self.relu(self.convd8x(res8x))
        res8x  = F.interpolate(res8x, res4x.size()[2:], mode='bilinear', align_corners=True)
        # res4x  = torch.add(res8x, res4x)
        res4x = torch.cat((res8x, res4x), 1)

        res4x  = self.relu(self.convd4x(res4x))
        res4x  = F.interpolate(res4x, res2x.size()[2:], mode='bilinear', align_corners=True)
        # res2x  = torch.add(res4x, res2x)
        res2x = torch.cat((res4x, res2x), 1)

        res2x  = self.relu(self.convd2x(res2x))
        res2x  = F.interpolate(res2x, x.size()[2:], mode='bilinear', align_corners=True)
        # x = torch.add(res2x, x)
        x = torch.cat((res2x, x), 1)

        x = self.conv_output(x)

        return x

def dimension_testing():
    model = nn.Sequential(
        MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ImproveNet(),
        InverseMeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    print(model)

    x = torch.rand(size=(16, 3, 640, 640), dtype=torch.float32).to(device)
    y = model(x)

    return True

def main():
    dimension_testing()

if __name__ == "__main__":
    main()
