#!/usr/bin/env python
"""
  FileName     [ psnr_ssim.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ (...) ]
"""

import argparse

import numpy as np
import scipy.misc
import skimage
from PIL import Image

import utils

"""
75 pth
rp: 6 PSNR: 22.6392712102
"""

def psnr_ssim(img_dehaze: Image, img_gt):
    dehaze = scipy.misc.fromimage(img_dehaze).astype(float) / 255.0
    gt     = scipy.misc.fromimage(img_gt).astype(float) / 255.0

    psnr = skimage.measure.compare_psnr(dehaze, gt)
    ssim = skimage.measure.compare_ssim(dehaze, gt, multichannel=True)
    
    return psnr, ssim

def val(dehazes, gts, outputpath):  
    psnrs = []
    ssims = []  
    
    # Write the psnr/ssim in the folder.
    with open(outputpath, "w") as textfile:
        for _, (dehaze, gt) in enumerate(zip(dehazes, gts)):
            print("{} / {}".format(dehaze, gt))
            
            img_dehaze, img_gt = Image.open(dehaze), Image.open(gt)
            psnr, ssim = psnr_ssim(img_dehaze, img_gt)
            psnrs.append(psnr)
            ssims.append(ssim)
            print("PSNR: {:.4f}".format(psnr))
            print("SSIM: {:.4f}".format(ssim))

            textfile.write(dehaze.split("/")[-1])            
            textfile.write("PSNR: {:.4f}, SSIM: {:.4f}\n".format(psnr, ssim))

        print("Average PSNR: {:.4f}".format(np.mean(psnrs)))
        print("Average SSIM: {:.4f}".format(np.mean(ssims)))
        textfile.write("Average PSNR: {:.4f}, Average SSIM: {:.4f}\n".format(np.mean(psnrs), np.mean(ssims)))

    return
    
def main():
    parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
    parser.add_argument("--dehaze", type=str, default="/media/disk1/EdwardLee/Output/", help="path to load dehaze images")
    parser.add_argument("--gt", type=str, default="/media/disk1/EdwardLee/IndoorTest/gt", help="path to load gt images")
    parser.add_argument("--output", type=str, default="./psnr_ssim.txt")

    opt = parser.parse_args()
    print(opt)

    dehazes = utils.load_all_image(opt.dehaze).sort()
    gts     = utils.load_all_image(opt.gt).sort()

    val(dehazes, gts, opt.output)

if __name__ == "__main__":
    main()
