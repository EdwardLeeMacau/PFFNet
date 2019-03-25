import argparse
import utils
import os

parser = argparse.ArgumentParser(description="Photo Combining")
parser.add_argument("--output", type=str, default="./output", help="path to save output images")
parser.add_argument("--gt", type=str, default="./dataset/reside/SOTS/indoor/gt", help="path of the ground truth images")
parser.add_argument("--wait", type=bool, action="store_true", help="if wait, stop when every photo comes in")
parser.add_argument("--verbose", type=str, action="store_true", help="increase the information verbosity")

opt = parser.parse_args()

def main():
    gtImages     = os.listdir(opt.gt)
    dehazeImages = os.listdir(opt.output)

if __name__ == "__main__":
    main()