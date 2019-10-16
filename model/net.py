"""
  FileName     [ net.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Basic units of the neural networks ]

  Layers:
    1. ConvLayer
    2. UpsampleConvLayer
    3. ResidualBlocks (BasicBlocks in Residual Net)
"""

import torch
import torch.nn as nn

class MeanShift(nn.Module):
    """ 
    Individual Module to Shift the batchSize

    Parameters
    ----------
    mean : 1-D tensor
    std : 1-D tensor
    """
    def __init__(self, mean, std, shape):
        if shape not in (3, 4):
            raise ValueError

        super(MeanShift, self).__init__()

        self.mean  = torch.tensor(mean, dtype=torch.float32).cuda()
        self.std   = torch.tensor(std, dtype=torch.float32).cuda()
        self.shape = shape

    def forward(self, x):
        if self.shape == 4:
            return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        if self.shape == 3:
            return (x - self.mean[:, None, None]) / self.std[:, None, None]

class ConvLayer(nn.Module):
    """ 
    Self defined convolutional layer class

    Features:
    - Use reflection padding instead of zero padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None):
        super(ConvLayer, self).__init__()

        layers = [
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        self.conv2d = nn.Sequential(*layers)

        # self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        # self.conv2d         = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # if norm_layer is not None:
        #     self.batchnorm  = norm_layer(outchannels)

    def forward(self, x):
        # out = self.reflection_pad(x)
        # out = self.conv2d(out)
        # return out

        return self.conv2d(x)


class UpsampleConvLayer(torch.nn.Module):
    """
    Self defined convolutional transpose layer class

    Features:
    - Use reflection padding instead of zero padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None):
        super(UpsampleConvLayer, self).__init__()

        layers = [
            nn.ReflectionPad2d(kernel_size // 2),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        self.convtranspose2d = nn.Sequential(*layers)

        # self.reflection_pad = nn.ReflectionPad2d(kernal_size // 2)
        # self.conv2d         = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

        # if norm_layer is not None:
        #     self.batchnorm  = norm_layer(outchannels)

    def forward(self, x):
        # out = self.reflection_pad(x)
        # out = self.conv2d(out)
        # return out

        return self.convtranspose2d(x)

class ResidualBlock(torch.nn.Module):
    """ 
    Self Defined Residual Blocks

    Feature:
    - Use self defined ConvLayer(Reflection Padding) Here
    """
    def __init__(self, channels, ratio=0.1, activation=nn.PReLU(), norm_layer=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, norm_layer=norm_layer)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, norm_layer=norm_layer)
        self.relu  = activation
        self.ratio = ratio

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out) * self.ratio
        out = torch.add(out, residual)

        return out

""" Still Developing """
class Bottleneck(torch.nn.Module):
    """
    Self Defined BottleNeck

    Features
    - Use self defined ConvLayer(Reflection Padding) Here
    """
    def __init__(self, channels, ratio=0.1, activation=nn.PReLU(), norm_layer=None):
        super(Bottleneck, self).__init__()

        self.relu  = activation

        self.conv1 = ConvLayer(channels, channels, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.bn1   = norm_layer(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, norm_layer=norm_layer)
        self.bn2   = norm_layer(channels)
        self.conv3 = ConvLayer(channels, channels, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.bn3   = norm_layer(channels)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(x)) * ratio

        out += identity
        out = self.relu(out)

        return out

""" Still Developing """
class AggregatedRecurrentResidualDownBlock(torch.nn.Module):
    def __init__(self, channels, activation=nn.PReLU()):
        super(AggregatedRecurrentResidualDownBlock, self).__init__()

        self.conv = ConvLayer(channels, channels, kernel_size=1, stride=1)
        self.relu = activation

    def forward(self, x):
        out = self.relu(self.conv(x))

        return out

""" Still Developing """
class AggregatedRecurrentResidualUpBlock(torch.nn.Module):
    def __init__(self, channels, activation=nn.PReLU()):
        super(AggregatedRecurrentResidualUpBlock, self).__init__()

        self.conv = UpsampleConvLayer(channels, channels, kernel_size=1, stride=1)
        self.relu = activation

    def forward(self, x):
        out = self.relu(self.conv(x))

        return out

def dimension_testing():
    return

def main():
    dimension_testing()

if __name__ == "__main__":
    main()
