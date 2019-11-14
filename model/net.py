"""
  FileName     [ net.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Basic units of the neural networks ]
"""

import torch
import torch.nn as nn

class MeanShift(nn.Conv2d):
    """ 
    Individual Module to Shift the batch of Images

    Parameters
    ----------
    mean, std : 1-D tensor
        (...)
    """
    def __init__(self, mean, std):
        super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)

        std = torch.Tensor(std)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data   = torch.Tensor(mean) / std

        self.requires_grad = False

class InverseMeanShift(nn.Conv2d):
    """
    Individual Module to Shift the batch of Images

    Parameters
    ----------
    mean, std : 1-D tensor
        (...)
    """
    def __init__(self, mean, std):
        super(InverseMeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)

        std = torch.Tensor(std)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1) * std.view(3, 1, 1, 1)
        self.bias.data   = torch.Tensor(mean)

        self.requires_grad = False

class AbstractConvLayer(nn.Module):
    """
    - Conv2d(in_channels, out_channels, kernel_size, stride=1)
    - ConvTransposs2d(in_channels, out_channels, kernel_size, stride=1)
    """
    def __init__(self, conv_layer, in_channels, out_channels, kernel_size, 
                 stride, norm_layer):
        super(AbstractConvLayer, self).__init__()

        layers = [
            nn.ReflectionPad2d(kernel_size // 2),
            conv_layer(in_channels, out_channels, kernel_size, stride),
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        self.conv2d = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv2d(x)

class ConvLayer(AbstractConvLayer):
    """ 
    Self-define convolutional layer class

    Output Dimension:
        (
            batchsize,
            out_channel,
            (Height + kernel_size // 2) / stride + 1,
            (Width + kennel_size // 2) / stride + 1
        )

    Example:
        Input Dimension:
            (batchsize, 16, 320, 320)
        Output Dimension:
            (batchsize, out_channel, (Height + (kernel_size // 2)) / stride + 1, (Width + (kernel_size // 2)) / stride + 1)
            = (batchsize, 32, 160, 160)

    Features:
    - Use reflection padding instead of zero padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, norm_layer):
        super(ConvLayer, self).__init__(
            nn.Conv2d, in_channels, out_channels, kernel_size, stride, norm_layer
        )
    # def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None):
    #     super(ConvLayer, self).__init__()

    #     layers = [
    #         nn.ReflectionPad2d(kernel_size // 2),
    #         nn.Conv2d(in_channels, out_channels, kernel_size, stride),
    #     ]

    #     if norm_layer is not None:
    #         layers.append(norm_layer(out_channels))

    #     self.conv2d = nn.Sequential(*layers)

    # def forward(self, x):
    #     return self.conv2d(x)

class UpsampleConvLayer(AbstractConvLayer):
    """
    Self-define convolutional transpose layer

    Output Dimension:
        (
            batchsize, 
            out_channels, 
            height * stride - 2 * padding(0) + kernel_size + output_padding(0), 
            width * stride - 2 * padding(0) + kernel_size + output_padding(0) 
        )

    Example:
        Input Dimension:  
            (batchsize, in_channels, 40, 40)
        Output Dimension: 
            (batchsize, out_channels, (40 + (3 // 2) * 2 - 1) * 2 - 0 + 3 + 0, (40 + (3 // 2) * 2 - 1) * 2 - 0 + 3 + 0)
            = (batchsize, out_channels, 85, 85)

    Features:
    - Use reflection padding instead of zero padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, norm_layer=None, interpolated=None):
        super(UpsampleConvLayer, self).__init__(
            nn.ConvTranspose2d, in_channels, out_channels, kernel_size, stride, norm_layer
        )

    # def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None, interpolated=None):
    #     super(UpsampleConvLayer, self).__init__()

    #     layers = [
    #         nn.ReflectionPad2d(kernel_size // 2),
    #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
    #     ]

    #     if norm_layer is not None:
    #         layers.append(norm_layer(out_channels))

    #     self.convtranspose2d = nn.Sequential(*layers)

    # def forward(self, x):
    #     return self.convtranspose2d(x)

""" Still Developing """
class AbstractResidualBlock(nn.Module):
    def __init__(self, conv_layer, in_channels, out_channels, ratio, kernel_size, stride, 
                 activation, norm_layer):
        super(AbstractResidualBlock, self).__init__()

        self.conv1 = conv_layer(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride, norm_layer=norm_layer
        )
        self.conv2 = conv_layer(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=1, norm_layer=norm_layer
        )
        self.relu  = activation
        self.ratio = ratio

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.ratio * self.conv2(out)
        out = torch.add(out, residual)

        return out

class ResidualBlock(torch.nn.Module):
    """ 
    Self-defined Residual Blocks

    Features:
    - Use self-define ConvLayer(Reflection Padding)
    - Stride is 1 for second layer
    """
    def __init__(self, in_channels, out_channels, ratio=0.1, kernel_size=3, stride=1,
                 activation=nn.PReLU(), norm_layer=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvLayer(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=stride, norm_layer=norm_layer
        )
        self.conv2 = ConvLayer(
            out_channels, out_channels, 
            kernel_size=kernel_size, stride=1, norm_layer=norm_layer
        )
        self.relu  = activation
        self.ratio = ratio

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.ratio * self.conv2(out)
        out = torch.add(out, residual)

        return out

""" Still Developing """
class UpsampleResidualBlock(ResidualBlock):
    """
    interploated = F.interpolate(res2x, x.size()[2:], mode='bilinear', align_corners=True)
    """
    def __init__(self, in_channels, out_channels, ratio=0.1, kernel_size=3, stride=1,
                 activation=nn.PReLU(), norm_layer=None, interpolated=None):
        super(UpsampleResidualBlock, self).__init__(in_channels, out_channels, ratio, kernel_size, stride, activation, norm_layer)

        self.conv1 = UpsampleConvLayer(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, norm_layer=norm_layer
        )
        self.conv2 = UpsampleConvLayer(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=1, norm_layer=norm_layer
        )

""" Still Developing """
class Bottleneck(torch.nn.Module):
    """
    Sel-define BottleNeck

    Features
    - Use self-define ConvLayer(Reflection Padding)
    """
    def __init__(self, channels, ratio=0.1, activation=nn.PReLU(), norm_layer=None):
        super(Bottleneck, self).__init__()

        self.relu  = activation
        self.ratio = ratio

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
    net = UpsampleResidualBlock(256, 128, 0.1, 3, 1)
    x   = torch.rand((16, 256, 40, 40))
    y   = net(x)
    return True

def main():
    dimension_testing()

if __name__ == "__main__":
    main()
