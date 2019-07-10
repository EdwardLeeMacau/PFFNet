#!/usr/bin/env python
"""
  FileName     [ psnr_ssim.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ (...) ]
"""

import argparse
import os

import numpy as np
import scipy.misc
import skimage
from PIL import Image

import utils

def psnr_ssim(img_dehaze: Image, img_gt: Image):
    dehaze = scipy.misc.fromimage(img_dehaze).astype(float) / 255.0
    gt     = scipy.misc.fromimage(img_gt).astype(float) / 255.0

    psnr = skimage.measure.compare_psnr(dehaze, gt)
    ssim = skimage.measure.compare_ssim(dehaze, gt, multichannel=True)
    
    return psnr, ssim

def val(dehazes, gts, output_path=None):
    """
      Validate the dehazing performance using PSNR and SSIM

      Params:
      - dehazes
      - gts
      - output_path

      Return: None
    """
    psnrs = []
    ssims = []  
    
    for _, (dehaze, gt) in enumerate(zip(dehazes, gts)):
        print("{} / {}".format(dehaze, gt))
        
        img_dehaze, img_gt = Image.open(dehaze), Image.open(gt)
        psnr, ssim = psnr_ssim(img_dehaze, img_gt)
        psnrs.append(psnr)
        ssims.append(ssim)
        print("PSNR: {:.4f}".format(psnr))
        print("SSIM: {:.4f}".format(ssim))

    print("Average PSNR: {:.4f}".format(np.mean(psnrs)))
    print("Average SSIM: {:.4f}".format(np.mean(ssims)))

    # Using np.lexsort(keys, axis) -> indices (ascending priority)
    if output_path is not None:
        nparray = np.array([psnrs, ssims])
        np.savetxt(output_path, nparray)

    return
    
def main():
    parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
    parser.add_argument("--dehaze", type=str, default="/media/disk1/EdwardLee/Output/", help="path to load dehaze images")
    parser.add_argument("--gt", type=str, default="/media/disk1/EdwardLee/IndoorTest/gt", help="path to load gt images")
    parser.add_argument("--output", type=str)

    opt = parser.parse_args()
    print(opt)

    dehazes = utils.load_all_image(opt.dehaze).sort()
    gts     = utils.load_all_image(opt.gt).sort()

    val(dehazes, gts, opt.output)

if __name__ == "__main__":
    main()
