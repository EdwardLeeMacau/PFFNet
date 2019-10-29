import os
import cv2
import numpy as np
import argparse

from PIL import Image
from functions import *

def deHaze(filename):
    fileName = os.path.basename(filename)
    print('Loading Image', filename)
    
    image    = Image.fromarray(readIm(filename))
    w, h     = image.size
    imageRGB = np.array(image)
    cv2.imwrite('MyResults/' + fileName + '_imageRGB.jpg', imageRGB)

    # Normalized as [0, 1]
    imageRGB = imageRGB / 255.0

    print('Getting Dark Channel Prior')
    darkChannel = getDarkChannel(imageRGB);
    cv2.imwrite('MyResults/' + fileName + '_dark.jpg', darkChannel * 255.0)

    print('Getting Atmospheric Light')
    atmLight = getAtmLight(imageRGB, darkChannel);

    print('Getting Transmission')
    transmission = getTransmission(imageRGB, atmLight);
    cv2.imwrite('MyResults/' + fileName + '_transmission.jpg', transmission * 255.0)

    print('Getting Scene Radiance')
    radiance = getRadiance(atmLight, imageRGB, transmission);
    cv2.imwrite('MyResults/' + fileName + '_radiance.jpg', radiance * 255.0)

    print('Apply Soft Matting')
    mattedTransmission = performSoftMatting(imageRGB, transmission);
    cv2.imwrite('MyResults/' + fileName + '_refinedTransmission.jpg', mattedTransmission * 255.0)

    print('Getting Scene Radiance')
    betterRadiance = getRadiance(atmLight, imageRGB, mattedTransmission);
    cv2.imwrite('MyResults/' + fileName + '_refinedRadiance.jpg', betterRadiance * 255.0)

    return betterRadiance

def main(args):
    deHaze(args.file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove image haze using dark channel prior method')
    parser.add_argument('-f', '--file', required=True, help="file name")
    parser.add_arugment('-o', '--output', help="Output directory")

    args = parser.parse_args()

    main(args)
