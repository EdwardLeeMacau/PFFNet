"""
  FileName    [ haze.py ]
  PackageName [ DarkChannelPrior ]
  Synposis    [  ]
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

from functions import *


def deHaze(filename, directory, refine=False):
    fileName = os.path.basename(filename)
    imageRGB = np.array(Image.fromarray(readIm(filename)))
    cv2.imwrite(os.path.join(directory, fileName + '_imageRGB.jpg'), imageRGB)

    # Normalized as [0, 1]
    imageRGB = imageRGB / 255.0

    print('Getting Dark Channel Prior')
    darkChannel = getDarkChannel(imageRGB)
    cv2.imwrite(os.path.join(directory, fileName + '_dark.jpg'), darkChannel * 255.0)

    print('Getting Atmospheric Light')
    atmLight = getAtmLight(imageRGB, darkChannel)

    print('Getting Transmission')
    transmission = getTransmission(imageRGB, atmLight)
    cv2.imwrite(os.path.join(directory, fileName + '_transmission.jpg'), transmission * 255.0)

    print('Getting Scene Radiance')
    radiance = getRadiance(atmLight, imageRGB, transmission)
    cv2.imwrite(os.path.join(directory, fileName + '_radiance.jpg'), radiance * 255.0)

    if refine:
        print('Apply Soft Matting')
        mattedTransmission = performSoftMatting(imageRGB, transmission)
        cv2.imwrite(os.path.join(directory, fileName + '_refinedTransmission.jpg'), mattedTransmission * 255.0)

        print('Getting Scene Radiance')
        betterRadiance = getRadiance(atmLight, imageRGB, mattedTransmission)
        cv2.imwrite(os.path.join(directory, fileName + '_refinedRadiance.jpg'), betterRadiance * 255.0)

        return betterRadiance

    return radiance

def main(args):
    deHaze(args.file, args.output, refine=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove image haze using dark channel prior method')
    parser.add_argument('-f', '--file', required=True, help="file name")
    parser.add_argument('-o', '--output', required=True, help="Output directory")

    args = parser.parse_args()

    main(args)
