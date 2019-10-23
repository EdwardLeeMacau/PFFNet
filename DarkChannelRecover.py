"""
  Filename     [ DrakChannelPrior.py ]  
  PackageName  [ PFFNet ]
  Synposis     [ a module for a dark channel based algorithm which remove haze on picture ]
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import numpy as np
from PIL import Image
import cv2

def brightness(img):
    """
    Get the brightness of the color image.

    Brightness(x) = Mean(Red(x), Green(x), Blue(x))

    Parameters
    ----------
    img : torch.tensor
        Tnesor in shape (batchSize, channel, height, width)

    Return
    ------
    intensity : torch.tensor
        Tensor in shape (batchSize, height, width)
    """
    return torch.mean(img, dim=1)

def intensity(img):
    """
    Get the intensity of color image.

    Remind: Intensity is not well defined in Image Processin.
            Using brightness here.

    Parameters
    ----------
    img : torch.tensor
        Tensor in shape (batchSize, channel, height, width)

    Return
    ------
    intensity : torch.tensor
        Tensor in shape (batchSize, height, width)
    """

    return brightness(img)
    # return torch.mean(img, dim=1)

def minPooling2d(tensor, kernel_size, max_value=255):
    """ 
    Local MinPooling Function for color Image

    Parameters
    ----------
    tensor : torch.tensor
        Tensor in shape (batchSize, channel, height, width)

    kernel_size : int
        The pooling size

    max_value : int, default=255
        The maximum value of the image

    Return
    ------
    localMin : torch.tensor
        Tensor in shape (batchSize, channel, height, width)
    """

    return max_value - F.max_pool2d(max_value - tensor, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

def getMinChannel(img):
    """ 
    Get the minimum value matrix (The minimum value in RGB Channel)

    Parameters
    ----------
    img : torch.tensor
        Tensor in shape (channel, height, width)

    Return
    ------
    imgGray : torch.tensor
        Tensor in shape (height, width)
    """
    if not (len(img.shape) == 3):
        raise ValueError("Image should have 3 dimensions")

    if not (img.shape[0] == 3):
        raise ValueError("Image should have 3 channels.")

    return torch.min(img, axis=0)[0]

def getDarkChannel(img, blockSize = 3):
    """ 
    Get the Dark Channel J^dark(x) 

    Parameters
    ----------
    img : torch.tensor
        3D or 4D tensor, (BatchSize, Channel, Height, Width)

    blockSize : int
        kernel size

    Return
    ------
    imgDark : ndarray
        Tensor in shape (BatchSize, Height, Width)
    """

    if len(img.shape) not in (3, 4):
        raise ValueError("The image shape should equal to 3 or 4(BatchMode)")

    if blockSize % 2 == 0:
        raise ValueError("The blocksize is not odd")

    if blockSize < 3:
        raise ValueError("The blocksize is too small")

    indice_channel = len(img.shape) - 3
    return torch.min(minPooling2d(img, kernel_size=blockSize, max_value=255), dim=indice_channel)[0]

def getAtmosphericLight(darkChannel, img, meanMode=False, percent=0.001):
    """ 
    Get AtomsphericLight A(c), the vector of size (3)

    Parameters
    ----------
    darkChannel : torch.tensor
        (batch, Height, Width)

    img : torch.tensor
        (batch, Channel, Height, Width)

    meanMode : bool
        Using mean mode to get lightMap

    percent : float
        (...)

    Return
    ------
    AtomsphericLight : array-like
        shape = (3, )
    """
    height, width = darkChannel.shape[-2:]
    size = height * width
    num  = np.ceil(percent * height * width).astype(np.int64).item()
    start_dim = 1

    # Find top k brighest pixels in dark channel.
    _, mask = torch.topk(torch.flatten(darkChannel, start_dim=1), k=num, dim=1)

    # Find the pixel with highest intensity
    _, index = torch.max(torch.flatten(intensity(img), start_dim=1)[0, mask], dim=1)

    # Return the highest intensity pixel
    return img.flatten(start_dim=2)[:, :, index].view(-1, 3)

def getTransmissionMap(img, omega, atmosphericLight, blockSize):
    """
    Getr transmission Map t(x), by t(x) = 1 - omega * min( min( I^{c}(y) / A^{c} ) )

    Parameters
    ----------
    img : torch.tensor
        (...)

    atomsphericLight : torch.tensor
        (...)

    omega : float
        (...)

    blockSize : float
        (...)

    Return
    ------
    transmission : torch.Tensor
        Tensor in shape (batchSize, height, width)
    """
    localMin = minPooling2d(img, kernel_size=blockSize, max_value=255)
    transmission = 1 - omega * torch.min(localMin / atmosphericLight[:, :, None, None], dim=1)[0]

    return transmission

def softMatting(imgMap: torch.Tensor):
    """
    Applying soft matting algorithm

    Parameters
    ----------
    imgMap : torch.tensor
        Tensor in shape (batchSize, height, width)

    Return
    ------
    matted_imgMap : torch.tensor
        Tensor in shape (batchSize, height, width)
    """

    return

def getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001):
    """ 
    Get recovered J(x) = (I(x) - A) / max(t(x), t0) + A

    Parameters
    ----------
    img : torch.tensor
        (batchSize, channel, heigth, width

    omega : float
        the proportion of dehazing

    t0 : float
        minimum transmission valu

    blockSize : int
        kernel size

    meanMode : bool
        If True, use the mean of RGB channel instead of the maximum value.

    precent : float
        (...)
    """
    # Dark Channel J^{dark}(x)
    imgDark = getDarkChannel(img, blockSize=blockSize)

    # AtmosphericLight A(c)
    atmosphericLight = getAtmosphericLight(imgDark, img, meanMode=meanMode, percent=percent)

    # Transmission t(x), minimum value of t(x) is t0
    transmission = torch.clamp(getTransmissionMap(img, omega, atmosphericLight, blockSize), min=t0)
    
    # Recovered image J(x)
    sceneRadiance = (img - atmosphericLight[:, :, None, None]) / transmission[:, None, :, :] + atmosphericLight[:, :, None, None]
    sceneRadiance = torch.clamp(sceneRadiance, min=0, max=255)

    return imgDark, transmission, sceneRadiance

def sample(image, output):
    """ Example of DarkChannelPrior Transform. """
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = torch.unsqueeze(torch.from_numpy(img).type(torch.float).permute(2, 0, 1), dim=0)

    imgDark, transmission, sceneRadiance = getRecoverScene(img)
    imgDark       = imgDark[0].type(torch.uint8).numpy()
    transmission  = (255 * transmission[0]).type(torch.uint8).numpy()
    sceneRadiance = sceneRadiance[0].permute(1, 2, 0).type(torch.uint8).numpy()

    if not output is None:
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)

        cv2.imwrite(os.path.join(output, 'dark.png'), imgDark)
        cv2.imwrite(os.path.join(output, 'trans.png'), transmission)
        cv2.imwrite(os.path.join(output, 'dcp.png'), sceneRadiance)

def main(args):
    sample(args.img, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--img", required=True)
    parser.add_argument('-o', "--output", help='Output Directory')
    args = parser.parse_args()

    main(args)
