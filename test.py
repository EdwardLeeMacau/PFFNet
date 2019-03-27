#!/usr/bin/env python
import argparse
import torch
import utils
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize
import os

"""
  Some notes:
  1. In pyTorch-1.0.1, torch.autograd.Variable is deprecated, the function Variable(tensor) return tensors
  2. In pyTorch-0.4.0, numpy is doesn't support problem.
"""

verbose = False

from model.rpnet import Net

parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--rb", type=int, default=13, help="number of residual blocks")
parser.add_argument("--checkpoint", type=str, default="./model/I-HAZE_O-HAZE.pth", help="path to load model checkpoint")
parser.add_argument("--test", type=str, default="./dataset/reside/SOTS/indoor/hazy", help="path to load test images")
parser.add_argument("--output", type=str, default="./output", help="path to save output images")
parser.add_argument("--verbose", action="store_true", help="increase the information verbosity")

opt = parser.parse_args()
print(opt)

"""
  Setting: 
  1. Default using cuda in test.py
  2. Read I-HAZE_O-HAZE as model.
  3. Only GPU-Cuda:0 is used in lab507
"""

# rb: Residual Blocks
net = Net(opt.rb)
net.load_state_dict(torch.load(opt.checkpoint)['state_dict'])
net.eval()
net = nn.DataParallel(net, device_ids=[0]).cuda()
print(net)

# Test photos: Default Reside
images = utils.load_all_image(opt.test)

# Output photos
if not os.path.exists(opt.output):
    os.mkdir(opt.output)

# Ignore .keep for folder
if ".keep" in images:
    images.remove(".keep")

for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    im = Image.open(im_path)
    h, w = im.size
    
    if verbose:
        print("Input filename: {}".format(filename))
        print("Image shape: {}, {}".format(h, w))
    
    im = ToTensor()(im)
    im = Variable(im).view(1, -1, w, h)
    im = im.cuda()
    
    with torch.no_grad():
        im = net(im)
    
    # Take the Image out from GPU.
    im = torch.clamp(im, 0., 1.)
    im = im.cpu()
    im = im.data[0]
    im = ToPILImage()(im)

    im.save(os.path.join(opt.output, filename))
