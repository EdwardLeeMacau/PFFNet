import argparse
import utils
import os

parser = argparse.ArgumentParser(description="Photo Combining")
parser.add_argument("--dehazy", type=str, default="./output/NTIRE2018/indoor", help="path to save dehazy images")
parser.add_argument("--gt", type=str, default="./IndoorTrainGT", help="path of the ground truth images")
parser.add_argument("--hazy", type=str, default="./IndoorTrainHazy", help="path of the haze images")
parser.add_argument("--merge", type=str, default="./merge", help="path to save the output images")
parser.add_argument("--wait", action="store_true", help="if wait, stop when every photo comes in")
parser.add_argument("--verbose", action="store_true", help="increase the information verbosity")

opt = parser.parse_args()

def main():
    gtImages     = os.listdir(opt.gt)
    hazyImages   = os.listdir(opt.hazy)
    dehazyImages = os.listdir(opt.dehazy)

    """ RESIDE: Multipul hazy images come from one gt image. """
    """ NTIRE2018: 1 GT image to 1 hazy image to 1 dehazy image. """
    for gt_image in gtImages:
        gt_image = gt_image[:gt_image.find("_")]

    for dehazy_image in dehazyImages:
        dehazy_image = dehazy_image[:dehazy_image.find("_")]
   
    for hazy_image in hazyImages:
        hazy_image = hazy_image[:hazy_image.find("_")]

    list.sort(gtImages)
    list.sort(dehazyImages)
    list.sort(hazyImages)

    for dehazy_image in dehazyImages:
        if dehazy_image in hazyImages:
            utils.combine_photo([os.path.join(opt.dehazy, dehazy_image), os.path.join(opt.hazy, dehazy_image)], os.path.join(opt.merge, dehazy_image))
            print("Combined: {}".format(dehazy_image))

if __name__ == "__main__":
    main()