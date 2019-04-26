"""
  library:
  * PIL     pip3 install pillow
  * scipy   pip3 install scipy
"""

import os
import numpy as np
from PIL import Image
import argparse
from scipy.misc import imsave
from scipy.ndimage import rotate
from joblib import Parallel, delayed

def augments(sp):
    print("Process %s" % sp)

    count_im = 0
    img_hazy = os.path.join(args.hazy, sp)
    img_gt   = os.path.join(args.gt, '_'.join([sp.split('_')[0], sp.split('_')[1], 'GT' + '.' + sp.split('_')[-1].split('.')[-1]]))

    for flip in [0, 1, 2]:
        for degree in [0, 1, 2, 3]:

            im_A = np.asarray(Image.open(img_hazy))
            im_B = np.asarray(Image.open(img_gt))

            # Horizontal / Vertial Flip
            if flip == 1:
                im_A = np.flip(im_A, 0)
                im_B = np.flip(im_B, 0)
            if flip == 2:
                im_A = np.flip(im_A, 1)
                im_B = np.flip(im_B, 1)

            # Rotation
            if degree != 0:
                im_A = rotate(im_A, 90 * degree)
                im_B = rotate(im_B, 90 * degree)

            # Height, Width, Channel
            h, w, c = im_A.shape

            for x in range(0, h, args.stride):
                for y in range(0, w, args.stride):

                    if x + args.size < h and y + args.size < w:
                        patch_A = im_A[x:x + args.size, y:y + args.size]    # Hazy images
                        patch_B = im_B[x:x + args.size, y:y + args.size]    # Clear images

                        imsave(os.path.join(args.output, "hazy", str(count_im).zfill(4) + "_" + "_".join(sp.split('_')[:-1]) + ".png"), patch_A)
                        imsave(os.path.join(args.output, "gt", str(count_im).zfill(4) + "_" + "_".join(sp.split('_')[:-1]) + ".png"), patch_B)
                        # imsave("%s/data/%d_%s.png" % (args.output, str(count_im).zfill(4), '_'.join(sp.split('_')[:-1])), patch_A)
                        # imsave("%s/label/%d_%s.png" % (args.output, str(count_im).zfill(4), '_'.join(sp.split('_')[:-1])), patch_B)

                        count_im += 1

    print("Process %s for %d" % (sp, count_im))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Data augmentation: Create image pairs')
    parser.add_argument("--size", type=int, default=512, help="which size to generate")
    parser.add_argument("--stride", type=int, default=256, help="the stride when cropping images")
    subparser  = parser.add_subparsers(required=True, dest="command")

    # I-Haze
    ihazeparser = subparser.add_parser("I-Haze")
    ihazeparser.add_argument('--hazy', help='Input directory for Hazy Image', type=str,
                            default="IndoorTrainHazy")
    ihazeparser.add_argument('--gt', help='Input directory for Clear Image', type=str,
                            default="IntdoorTrainGT")
    ihazeparser.add_argument('--output', help='Output directory', type=str,
                            default="IntdoorTrain")

    # O-Haze
    ohazeparser = subparser.add_parser("O-Haze")
    ohazeparser.add_argument('--hazy', help='Input directory for Hazy Image', type=str,
                            default="OutdoorTrainHazy")
    ohazeparser.add_argument('--gt', help='Input directory for Clear Image', type=str,
                            default="OutdoorTrainGT")
    ohazeparser.add_argument('--output', help='Output directory', type=str,
                            default="OutdoorTrain")
    args = parser.parse_args()

    print(args)

    splits = os.listdir(args.hazy)

    # Check folders here, make the directories if don't exist.
    # 1.  Root folder
    #     1.1 gt
    #     2.2 hazy
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if not os.path.exists(os.path.join(args.output, "gt")):
        os.makedirs(os.path.join(args.output, "gt"))
    else:
        os.system("rm " + os.path.join(args.output, "gt", "*"))

    if not os.path.exists(os.path.join(args.output, "hazy")):
        os.makedirs(os.path.join(args.output, "hazy"))
    else:
        os.system("rm " + os.path.join(args.output, "hazy", "*"))

    if not os.path.exists(args.hazy):
        raise IOError("File doesn't not exist: {}".format(args.hazy))

    if not os.path.exists(args.gt):
        raise IOError("File doesn't not exist: {}".format(args.gt))

    # Function parallel working...
    Parallel(-1)(delayed(augments)(sp) for sp in splits)
