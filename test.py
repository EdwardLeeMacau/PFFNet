#!/usr/bin/env python
"""
  FileName     [ test.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Generated the dehazed images from the PFFNet Model ]
"""

import argparse
import os

import torch
import utils
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize

from model.rpnet import Net

"""
  Some notes:
  1. In pyTorch-1.0.1, torch.autograd.Variable is deprecated, the function Variable(tensor) return tensors
  2. In pyTorch-0.4.0, numpy is doesn't support problem.
"""

def predict():
    parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
    parser.add_argument("--rb", type=int, default=18, help="number of residual blocks")
    parser.add_argument("--checkpoint", type=str, default="/media/disk1/EdwardLee/checkpoints/Indoor_18_16", help="root of model checkpoint")
    parser.add_argument("--pth", type=str, help="choose the checkpoint")
    parser.add_argument("--test", type=str, default="/media/disk1/EdwardLee/IndoorTest", help="path to load test images")
    parser.add_argument("--cuda", default=True, help="Use cuda?")
    parser.add_argument("--gpus", type=int, default=4, help="nums of gpu to use")
    parser.add_argument("--output", type=str, default="/media/disk1/EdwardLee/Output", help="path to save output images")
    parser.add_argument("--verbose", default=True, help="increase the information verbosity")

    # subparser = parser.add_subparsers(required=True, dest="command", help="I-Haze / O-Haze / SOTS")

    # ihazeparser = subparser.add_parser("I-Haze")
    # ihazeparser.add_argument("--checkpoint", default="/media/disk1/EdwardLee/checkpoints/Indoor_18_16", type=str, help="path to load train datasets")
    # ihazeparser.add_argument("--test", default="/media/disk1/EdwardLee/IndoorTest", type=str, help="path to load test datasets")

    # ohazeparser = subparser.add_parser("O-Haze")
    # ohazeparser.add_argument("--checkpoint", default="/media/disk1/EdwardLee/checkpoints/Outdoor_18_16", type=str, help="path to load train datasets")
    # ohazeparser.add_argument("--test", default="/media/disk1/EdwardLee/OutdoorTest", type=str, help="path to load test datasets")

    # sotsparser = subparser.add_parser("SOTS")
    # sotsparser.add_argument("--checkpoint", default="/media/disk1/EdwardLee/checkpoints/Indoor_18_16", type=str, help="path to load train datasets")
    # sotsparser.add_argument("--test", default="/media/disk1/EdwardLee/dataset/reside/SOTS/indoor", type=str, help="path to load test datasets")

    opt = parser.parse_args()
    print(opt)
    
    """
    Setting: 
    1. Read I-HAZE_O-HAZE as model.
    2. Only GPU-Cuda:0 is used in lab507
    """

    device = utils.selectDevice()

    checkpoint = os.path.join(opt.checkpoint, opt.pth)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exists: {}".format(checkpoint))

    # rb: Residual Blocks
    net = Net(opt.rb).to(device)
    net.load_state_dict(torch.load(checkpoint)['state_dict'])
    net.eval()
    # net = nn.DataParallel(net, device_ids=[0]).cuda()
    # print(net)

    # if opt.cuda:
    #     net = nn.DataParallel(net, device_ids=[i for i in range(opt.gpus)]).cuda()
    # else:
    #     net = net.cpu()
        
    # Test photos: Default Reside
    images = utils.load_all_image(opt.test)

    # Output photos
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    # Ignore .keep for folder
    if ".keep" in images:
        images.remove(".keep")

    print("==========> DeHazing, Target: {}".format(len(images)))
    for im_path in tqdm(images):
        filename = im_path.split('/')[-1]
        im = Image.open(im_path)
        h, w = im.size
        
        if opt.verbose:
            print("==========> Input filename: {}".format(filename))
            print("==========> Image shape: {}, {}".format(h, w))
        
        im = ToTensor()(im)
        im = im.view(1, -1, w, h).to(device)
        # im = im.cuda()
        
        with torch.no_grad():
            im_dehaze = net(im)
        
        # Take the Image out from GPU.
        im_dehaze = torch.clamp(im_dehaze, 0., 1.).cpu()
        im_dehaze = ToPILImage()(im_dehaze.data[0])

        im_dehaze.save(os.path.join(opt.output, filename))
        print("==========> File saved: {}".format(os.path.join(opt.output, filename)))

def main():
    predict()

if __name__ == "__main__":
    main()