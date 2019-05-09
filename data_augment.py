"""
  library:
  * PIL == 5.4.1    pip3 install pillow
  * scipy           pip3 install scipy
"""

import argparse
import os
import sys
import copy

import random
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from scipy.misc import imsave
from scipy.ndimage import rotate

def random_augments(sp):
    print("Process %s" % sp)

    count_im = 0
    img_hazy = os.path.join(args.hazy, sp)
    img_gt   = os.path.join(args.gt, '_'.join([sp.split('_')[0], sp.split('_')[1], 'GT' + '.' + sp.split('_')[-1].split('.')[-1]]))

    raw_im_A, raw_im_B = Image.open(img_hazy), Image.open(img_gt)

    w, h = raw_im_A.size
    l = min(h, w)

    # -----------------------------
    # Random crop from any coordinate
    #   Size: Random choice from list
    #   X: Uniform Choose
    #   Y: Uniform Choose
    #   Flip and 
    # -----------------------------

    for _ in range(args.location):
        # --------------------------------------------------------------------
        # Random choose size
        #   if image size is smaller than 1024, only 512 is cropped.
        #   if image_size is smaller than 1536, randomly choose 512 or 1024 as size
        #   if image_size is larger than 1536, randomly choose 512, 1024 or 1536
        # ----------------------------------------------------------------------
        if l < 1024:   
            size = 512
        elif l < 1536:      
            size = np.random.choice([512, 1024], size=None, replace=False, p=[0.6, 0.4])
        else:           
            size = np.random.choice([512, 1024, 1536], size=None, replace=False, p=[0.6, 0.3, 0.1]) 

        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        
        im_A, im_B = raw_im_A.copy(), raw_im_B.copy()
        im_A = im_A.crop((x, y, x+size, y+size)).resize((512, 512), Image.BILINEAR) # Hazy images
        im_B = im_B.crop((x, y, x+size, y+size)).resize((512, 512), Image.BILINEAR) # Clear images
        # print(0, 0, w, h)
        # print(x, y, x+size, y+size)

        flip = random.choice([0, 1, 2])
        degree = random.choice([0, 1, 2, 3])

        # Horizontal Flip: transforms.HorizontalFlip()
        # Vertial Flip: transforms.VerticalFlip()
        if flip == 1:
            im_A = im_A.transpose(Image.FLIP_LEFT_RIGHT)
            im_B = im_B.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip == 2:
            im_A = im_A.transpose(Image.FLIP_TOP_BOTTOM)
            im_B = im_B.transpose(Image.FLIP_TOP_BOTTOM)

        # Rotation: transforms.RandomRotation()
        if degree == 1:
            im_A = im_A.transpose(Image.ROTATE_90)
            im_B = im_B.transpose(Image.ROTATE_90)
        elif degree == 2:
            im_A = im_A.transpose(Image.ROTATE_180)
            im_B = im_B.transpose(Image.ROTATE_180)
        elif degree == 3:
            im_A = im_A.transpose(Image.ROTATE_270)
            im_B = im_B.transpose(Image.ROTATE_270)

        im_A.save(os.path.join(args.output, "hazy", str(count_im).zfill(4) + "_" + "_".join(sp.split('_')[:-1]) + ".png"))
        im_B.save(os.path.join(args.output, "gt", str(count_im).zfill(4) + "_" + "_".join(sp.split('_')[:-1]) + ".png"))
                
        count_im += 1
    
    print("Process %s for %d" % (sp, count_im))

def augments(sp):
    print("Process %s" % sp)

    count_im = 0
    img_hazy = os.path.join(args.hazy, sp)
    img_gt   = os.path.join(args.gt, '_'.join([sp.split('_')[0], sp.split('_')[1], 'GT' + '.' + sp.split('_')[-1].split('.')[-1]]))

    raw_im_A = np.asarray(Image.open(img_hazy))
    raw_im_B = np.asarray(Image.open(img_gt))

    for flip in [0, 1, 2]:
        for degree in [0, 1, 2, 3]:
            im_A = raw_im_A.copy()
            im_B = raw_im_B.copy()

            # Horizontal Flip: transforms.HorizontalFlip()
            if flip == 1:
                im_A = np.flip(im_A, 0)
                im_B = np.flip(im_B, 0)
            # Vertial Flip: transforms.VerticalFlip()
            if flip == 2:
                im_A = np.flip(im_A, 1)
                im_B = np.flip(im_B, 1)

            # Rotation: transforms.RandomRotation()
            if degree != 0:
                im_A = rotate(im_A, 90 * degree)
                im_B = rotate(im_B, 90 * degree)

            # Height, Width, Channel
            h, w, _ = im_A.shape

            for x in range(0, h, args.stride):
                for y in range(0, w, args.stride):

                    if x + args.size < h and y + args.size < w:
                        patch_A = im_A[x:x + args.size, y:y + args.size]    # Hazy images
                        patch_B = im_B[x:x + args.size, y:y + args.size]    # Clear images

                        imsave(os.path.join(args.output, "hazy", str(count_im).zfill(4) + "_" + "_".join(sp.split('_')[:-1]) + ".png"), patch_A)
                        imsave(os.path.join(args.output, "gt", str(count_im).zfill(4) + "_" + "_".join(sp.split('_')[:-1]) + ".png"), patch_B)
                        
                        count_im += 1

    print("Process %s for %d" % (sp, count_im))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Data augmentation: Create image pairs')
    parser.add_argument("--location", type=int, default=5000, help="how many location coordinate to sample")
    parser.add_argument("--size", type=int, default=512, help="the size when cropping images")
    parser.add_argument("--stride", type=int, default=256, help="the stride when cropping images")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--step", action="store_true")

    domainparser  = parser.add_subparsers(required=True, dest="domain")
    
    # I-Haze
    ihazeparser = domainparser.add_parser("I-Haze")
    ihazeparser.add_argument('--hazy', help='Input directory for Hazy Image', type=str,
                            default="/media/disk1/EdwardLee/dataset/IndoorTrainHazy")
    ihazeparser.add_argument('--gt', help='Input directory for Clear Image', type=str,
                            default="/media/disk1/EdwardLee/dataset/IndoorTrainGT")
    ihazeparser.add_argument('--output', help='Output directory', type=str,
                            default="/media/disk1/EdwardLee/IndoorTrain")

    # O-Haze
    ohazeparser = domainparser.add_parser("O-Haze")
    ohazeparser.add_argument('--hazy', help='Input directory for Hazy Image', type=str,
                            default="/media/disk1/EdwardLee/dataset/OutdoorTrainHazy")
    ohazeparser.add_argument('--gt', help='Input directory for Clear Image', type=str,
                            default="/media/disk1/EdwardLee/dataset/OutdoorTrainGT")
    ohazeparser.add_argument('--output', help='Output directory', type=str,
                            default="/media/disk1/EdwardLee/OutdoorTrain")

    args = parser.parse_args()
    print(args)

    # ----------------------------------------------------------
    # Check folders here, make the directories if don't exist.
    # 0.  The image
    # 1.  Root folder
    #     1.1 gt
    #     1.2 hazy
    # ---------------------------------------------------------
    if not os.path.exists(args.hazy):
        raise IOError("File doesn't not exist: {}".format(args.hazy))

    if not os.path.exists(args.gt):
        raise IOError("File doesn't not exist: {}".format(args.gt))

    # 1 Root Folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 1.1 Ground Truth
    if not os.path.exists(os.path.join(args.output, "gt")):
        os.makedirs(os.path.join(args.output, "gt"))
    else:
        print("Clean the folder: {}".format(os.path.join(args.output, "gt")))
        command = input("Clean the folder? [Y/N]")
        
        if command == 'Y':
            print("Clean {}.".format(os.path.join(args.output, "gt")))
            os.system("rm -r" + os.path.join(args.output, "gt"))
            os.makedirs(os.path.join(args.output, "gt"))
        
    # 1.2 Hazy Images
    if not os.path.exists(os.path.join(args.output, "hazy")):
        os.makedirs(os.path.join(args.output, "hazy"))
    else:
        print("Clean the folder: {}".format(os.path.join(args.output, "hazy")))
        command = input("Y/N")
        
        if command == 'Y':
            print("Clean {}".format(os.path.join(args.output, "hazy")))
            os.system("rm -r" + os.path.join(args.output, "hazy"))
            os.makedirs(os.path.join(args.output, "hazy"))
        
    # Function parallel working...
    splits = os.listdir(args.hazy)

    if args.step:
        print("Executing: augments")
        Parallel(-1)(delayed(augments)(sp) for sp in splits)
    elif args.random:
        print("Executing: random_augments")
        # Parallel(-1)(delayed(random_augments)(sp) for sp in splits)
        for sp in splits:
            random_augments(sp)
