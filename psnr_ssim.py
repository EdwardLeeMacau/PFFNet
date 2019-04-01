#!/usr/bin/env python
import argparse
import utils
from PIL import Image
import numpy as np
import scipy.misc
import skimage

parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--data", type=str, default="./output/NTIRE2018/indoor", help="path to load data images")
parser.add_argument("--gt", type=str, default="./IndoorTrainGT", help="path to load gt images")
parser.add_argument("--output", type=str, default="./psnr_ssim.txt")

opt = parser.parse_args()
print(opt)

datas = utils.load_all_image(opt.data)
gts = utils.load_all_image(opt.gt)

datas.sort()
gts.sort()

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

psnrs = []
ssims = []

with open(opt.output, "w") as textfile:
    for i in range(len(datas)):
        print("{} / {}".format(datas[i], gts[i]))
        textfile.write("{} / {}\n".format(datas[i], gts[i]))

        data = scipy.misc.fromimage(Image.open(datas[i])).astype(float)/255.0
        gt = scipy.misc.fromimage(Image.open(gts[i])).astype(float)/255.0

        psnr = skimage.measure.compare_psnr(data, gt)
        ssim = skimage.measure.compare_ssim(data, gt, multichannel=True)
        print("PSNR: {}".format(psnr))
        print("SSIM: {}".format(ssim))
        
        textfile.write("PSNR: {}\nSSIM: {}\n".format(psnr, ssim))
        psnrs.append(psnr)
        ssims.append(ssim)

    print("Average PSNR: {}\nAverage SSIM: {}".format(np.mean(psnrs), np.mean(ssims)))
    textfile.write("Average PSNR: {}\nAverage SSIM: {}".format(np.mean(psnrs), np.mean(ssims)))

"""
75 pth
rp: 6 PSNR: 22.6392712102
"""
