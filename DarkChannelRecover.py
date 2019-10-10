"""
  Filename [ DrakChannelPrior.py ]  
  Package  [ PFFNet ]
  Synposis [ a module for a dark channel based algorithm which remove haze on picture ]
"""

import argparse
import math
import numpy as np
from PIL import Image
import cv2

# 用于排序时存储原来像素点位置的数据结构
class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print('%s:%s:%s' %(self.x,self.y,self.value))

def getMinChannel(img):
    """ Get the minimum value matrix (The minimum value in RGB Channel) """
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        raise ValueError("Image should have 3 channels.")

    # img.shape = (H x W x C)
    whiteMap = np.ones((img.shape[0], img.shape[1], 1), dtype = np.uint8) * 255
    imgGray  = np.amin(np.concatenate((img, whiteMap), axis=2), axis=2)

    return imgGray

def getDarkChannel(img, blockSize = 3):
    """ Get the Dark Channel of the image """

    if len(img.shape) != 2:
        raise ValueError("The image shape should equal to 2")

    if blockSize % 2 == 0:
        raise ValueError("The blocksize is not odd")

    if blockSize < 3:
        raise ValueError("The blocksize is too small")

    # 计算addSize
    addSize = int((blockSize - 1) / 2)

    newHeight = img.shape[0] + blockSize - 1
    newWidth  = img.shape[1] + blockSize - 1

    # 中间结果
    # imgMiddle = np.repeat.padding(img)
    imgMiddle = np.ones((newHeight, newWidth, 1)) * 255
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize, 0] = img

    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    # Scaning Pixels without the image edge
    # TODO: Optimize the performance here
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            whiteMap = np.ones((blockSize, blockSize, 0)) * 255

            localMin = np.amin(np.concatenate((imgMiddle[i - addSize: i + addSize + 1, j - addSize : j + addSize + 1], whiteMap), axis=2), axis=None)
            imgDark[i - addSize, j - addSize] = localMin

    whiteMap = np.ones((blockSize, blockSize, 0)) * 255

    return imgDark

def getAtomsphericLight(darkChannel, img, meanMode=False, percent=0.001):
    """ 
    Get AtomsphericLight A(x)(1 - t(x)) 

    Parameters
    ----------
    darkChannel : array-like
        (...)

    img : array-like
        (...)

    meanMode : bool
        Using mean mode to get lightMap

    percent : float
        (...)

    Return
    ------
    AtomsphericLight : array-like
        (...)
    """
    size   = darkChannel.shape[0] * darkChannel.shape[1]
    height = darkChannel.shape[0]
    width  = darkChannel.shape[1]
    num    = np.ceil(percent * size).astype(np.int64).item()

    # Sorted nodes by darkChannel value in descending order
    # Only consider the N Nodes with the maximum darkChannel value
    # nodes = sorted([ Node(i, j, darkChannel[i, j]) for i in range(0, height) for j in range(0, width) ], key = lambda x: x.value, reverse=True)[:num]

    # Sorted nodes (sort by numpy)
    maxN = np.flip(np.argsort(darkChannel.flatten()))[:num]
    x = maxN // darkChannel.shape[1]
    y = np.mod(maxN, darkChannel.shape[1])

    atomsphericLight = 0
    
    # raise NotImplementedError

    # meanMode
    if meanMode:
        # sum = 0
        atomsphericLight = np.mean(img[x, y, :])
        # for i in range(0, num):
        #     for j in range(0, 3):
        #         sum = sum + img[nodes[i].x, nodes[i].y, j]
        #
        # atomsphericLight = int(sum / (int(percent * size) * 3))
	
        return atomsphericLight

    # 获取暗通道前0.1%(percent)的位置的像素点在原图像中的最高亮度值
    # 原图像像素过少时，只考虑第一个像素点
    atomsphericLight = np.amax(img[x, y, :])

    # for i in range(0, num):
    #     atomosphericLight = max([ atomosphericLight ].extend(img[nodes[i].x, nodes[i].y, :].item()))
    #     for j in range(0, 3):
    #         if img[nodes[i].x, nodes[i].y, j] > atomsphericLight:
    #             atomsphericLight = img[nodes[i].x, nodes[i].y, j]

    # print(atomsphericLight)

    return atomsphericLight

def getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001):
    """ 
    get recovered I(x) 

    Parameters
    ----------
    omega : float
        the proportion of dehazing

    t0 : float
        minimum transmission valu

    blockSize : int
        (...)

    meanMode : bool
        (...)

    precent : float
        (...)
    """
    imgGray = getMinChannel(img)
    imgDark = getDarkChannel(imgGray, blockSize=blockSize).astype(np.float64)
    # print(imgDark.shape)
    print("Getting atomsphericLight")
    # imgDark = np.random.randint(low=0, high=255, size=(2882, 4478))
    atomsphericLight = getAtomsphericLight(imgDark, img, meanMode=meanMode, percent=percent)

    # imgDark = np.float64(imgDark)

    # Calculate transmission t(x), minimum value of t(x) is t0
    transmission = 1 - omega * imgDark / atomsphericLight
    transmission = np.clip(transmission, a_min=t0, a_max=None)

    # Calculate recover image I(x)
    sceneRadiance = np.zeros(img.shape)

    print("Type of sceneRadiance: ", type(sceneRadiance))
    print("Shape of sceneRadiance: ", sceneRadiance.shape)

    for i in range(0, 3):
        img = np.float64(img)

        sceneRadiance[:, :, i] = (img[:, :, i] - atomsphericLight) / transmission + atomsphericLight
        sceneRadiance[:, :, i] = np.clip(sceneRadiance[:, :, i], a_min=0, a_max=255)

        # for j in range(0, sceneRadiance.shape[0]):
        #     for k in range(0, sceneRadiance.shape[1]):
        #         if sceneRadiance[j,k,i] > 255:
        #             sceneRadiance[j,k,i] = 255
        #         if sceneRadiance[j,k,i] < 0:
        #             sceneRadiance[j,k,i]= 0

    sceneRadiance = np.uint8(sceneRadiance)

    return sceneRadiance

def sample(image, output):
    """ Example of DarkChannelPrior Transform. """
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    # img = np.array(Image.open(image))
    sceneRadiance = getRecoverScene(img)

    if not output is None:
        if not os.path.exists(os.path.dirname(output)): 
            os.makedirs(output, exist_ok=True)

        cv.imwrite(output)

def main(args):
    sample(args.img, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--img", required=True)
    parser.add_argument('-o', "--output")
    args = parser.parse_args()

    main(args)
