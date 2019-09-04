"""
  FileName     [ preprocess.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Dataset augmentation function ]
"""

import argparse
import os
import random
import sys
import time
from joblib import Parallel, delayed

import numpy as np
from PIL import Image

from natsort import natsorted, ns

def main():
    # python3.6 Preprocess_train.py --inputHaze_dir ./sample/IHAZE_train_unprocessed --inputGT_dir ./sample/GT_train_unprocessed --trainData_dir ./sample/trainData
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputHaze_dir', default='/media/disk1/EdwardLee/IHAZE_train_unprocessed',  help='The root of hazy images')
    parser.add_argument('--inputGT_dir', default='/media/disk1/EdwardLee/GT_train_unprocessed',  help='The root of ground truths')
    parser.add_argument('--trainData_dir', default='/media/disk1/EdwardLee/trainData',  help='The root of output folder')
    
    parser.add_argument('--Scale_w', type=int, default=2560,  help='')
    parser.add_argument('--Scale_h', type=int, default=2560,  help='')
    parser.add_argument('--crop_w', type=int, default=640,  help='')
    parser.add_argument('--crop_h', type=int, default=640,  help='')
    parser.add_argument('--location_count', type=int, default=100,  help='')
    parser.add_argument('--index_start', type=int, default=0,  help='')

    opt = parser.parse_args()

    inputHaze_dir = opt.inputHaze_dir 
    inputGT_dir = opt.inputGT_dir 

    if not os.path.exists(inputHaze_dir):
        raise IOError("{} doesn't exists. ".format(inputHaze_dir))

    if not os.path.exists(inputGT_dir):
        raise IOError("{} doesn't exists. ".format(inputGT_dir))

    trainData_dir = opt.trainData_dir 
    os.makedirs(trainData_dir, exist_ok=True)
    os.makedirs(os.path.join(trainData_dir, "hazy"))
    os.makedirs(os.path.join(trainData_dir, "gt"))

    # eat training data as paired (IHAZE_train_unprocessed, GT_train_unprocessed) images seperately: imgA, imgB
    # transforms.Scale(originalSize),
    # transforms.RandomCrop(imageSize),
    # transforms.RandomHorizontalFlip(),
    # after 20 * 2(hazy and gt) original images with 100 different locations 640*640 small img cropped.  We have 2000 pairs of data.  
    # DONT COMBINE hazy and gt!!!! Time consuming...
    # save in /trainData as 0_hazy.png, 0_gt.png, 1_hazy.png, 1_gt.png......
    # ---------
    # pix2pix_notcombined will eat _hazy.png or _gt.png and output it to imgA, imgB

    fnames1 = []
    fnames2 = []
    
    for root1, _, fnames in (os.walk(inputHaze_dir)):
        for _, fname1 in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
            img1 = Image.open(os.path.join(inputHaze_dir, fname1)).convert('RGB') # img = Image.open(os.path.join(root, fname)).convert('RGB')
            fnames1.append(img1)
    
    for root2, _, fnames in (os.walk(inputGT_dir)):
        for _, fname2 in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):  
            img2 = Image.open(os.path.join(inputGT_dir, fname2)).convert('RGB') # img = Image.open(os.path.join(root, fname)).convert('RGB')
            fnames2.append(img2)

    index = opt.index_start
    t0 = time.time()

    # 100 different locations >> gives total outputs of 20*100=2000 pairs of (hazy, gt) imgs
    for i in range(0, opt.location_count): 
        for j, (imgA, imgB) in enumerate( zip(fnames1, fnames2) ):
            index = index + 1
            
            # transforms.Scale(originalSize),
            imgA = imgA.resize((opt.Scale_w, opt.Scale_h), Image.BILINEAR)
            imgB = imgB.resize((opt.Scale_w, opt.Scale_h), Image.BILINEAR)
            
            # transforms.RandomCrop(imageSize),
            w, h = imgA.size # imgAsize==imgBsize
            tw, th = opt.crop_w, opt.crop_h
            if not(w == tw and h == th):
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                imgA = imgA.crop((x1, y1, x1 + tw, y1 + th))
                imgB = imgB.crop((x1, y1, x1 + tw, y1 + th))

            # transforms.RandomHorizontalFlip(),
            flag_horizontal = random.random() < 0.5
            if flag_horizontal:
                imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
                imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)

            # transforms.RandomVerticalFlip()
            flag_vertical = random.random() < 0.5
            if flag_vertical:
                imgA = imgA.transpose(Image.FLIP_TOP_BOTTOM)
                imgB = imgB.transpose(Image.FLIP_TOP_BOTTOM)            

            # save all in trainData/hazy and trainData/gt
            imgA.save(os.path.join(trainData_dir, "hazy", str(index) + '_hazy.png'))
            imgB.save(os.path.join(trainData_dir, "gt", str(index) + '_gt.png'))

    t1 = time.time()
    print('Running time:'+str(t1-t0))

if __name__ == "__main__":
    main()
