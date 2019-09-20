#!/usr/bin/env python
"""
  FileName     [ psnr_ssim.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ (...) ]
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.misc
import skimage
from PIL import Image

import utils

def psnr_ssim(img_dehaze: Image, img_gt: Image):
    """
    Calculate PSNR and SSIM value for 2 Images

    Parameters
    ----------
    img_dehaze, img_gt : PIL.Image
        (...)

    Return
    ------
    psnr, ssim : np.float32
        PSNR and SSIM value of the image pair.
    """
    dehaze = np.asarray(img_dehaze).astype(float) / 255.0
    gt     = np.asarray(img_gt).astype(float) / 255.0

    psnr = skimage.measure.compare_psnr(dehaze, gt)
    ssim = skimage.measure.compare_ssim(dehaze, gt, multichannel=True)
    
    return psnr, ssim

def val(dehazes, gts, output_path=None):
    """
    Validate the dehazing performance using MSE, PSNR, SSIM

    Parameters
    ----------
    dehazes, gt : list-like
        The file name of dehazed images and ground truth images
    
    output_path : {str, None} optional
        If not None, save the PSNR and SSIM to the Textfile

    Return
    ------
    psnrs, ssims : np.float
        (....)
    """
    psnrs = []
    ssims = []
    index = []
    
    for _, (dehaze, gt) in enumerate(zip(dehazes, gts)):        
        img_dehaze, img_gt = Image.open(dehaze), Image.open(gt)
        psnr, ssim = psnr_ssim(img_dehaze, img_gt)

        psnrs.append(psnr)
        ssims.append(ssim)
        index.append(os.path.basename(dehaze).split("_")[0])

        print("GT: {}".format(gt))
        print("Dehaze: {}".format(dehaze))
        print("PSNR: {:.4f}".format(psnr))
        print("SSIM: {:.4f}".format(ssim))

    # Summary
    psnr_mean, psnr_std = np.mean(psnrs), np.std(psnrs)
    ssim_mean, ssim_std = np.mean(ssims), np.std(ssims)

    psnrs.append(psnr_mean)
    ssims.append(ssim_mean)
    index.append("Mean")

    psnrs.append(psnr_std)
    ssims.append(ssim_std)
    index.append("Std")

    print("Validate result: ")
    print("Average PSNR: {:.4f}, STD: {:.6f}".format(psnr_mean, psnr_std))
    print("Average SSIM: {:.4f}, STD: {:.6f}".format(ssim_mean, ssim_std))

    # Generate summary doc.
    if output_path is not None:
        nparray = np.array([psnrs, ssims])
        df = pd.DataFrame(data={'psnr': psnrs, 'ssim': ssims}, index=index)
        
        # Text Format
        np.savetxt(os.path.join(output_path, "record.txt"), nparray)

        # JSON Format
        df.to_json(os.path.join(output_path, "record.json"), orient='index')

        # Spreadsheet Format
        df.transpose().to_excel(os.path.join(output_path, "record.xlsx"))

    return psnrs, ssims, index
    
def main():
    dehazes = utils.load_all_image(opt.dehaze).sort()
    gts     = utils.load_all_image(opt.gt).sort()

    val(dehazes, gts, opt.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
    parser.add_argument("--dehaze", type=str, help="path to load dehaze images")
    parser.add_argument("--gt", type=str, default="./dataset/NTIRE2018_TEST/GT", help="path to load gt images")
    parser.add_argument("--output", type=str)

    opt = parser.parse_args()

    if opt.dehaze is None:
        raise ValueError("Please type in the dehazed images directory with --dehaze <directory>")

    if not os.path.exists(opt.dehaze):
        raise ValueError("Directory {} doesn't exists".format(opt.dehaze))

    if not os.path.exists(opt.gt):
        raise ValueError("Directory {} doesn't exists".format(opt.gt))

    utils.detail(opt, None)

    main(opt)
