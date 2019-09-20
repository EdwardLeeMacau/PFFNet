#!/usr/bin/env python
"""
  FileName     [ test.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Generated the dehazed images by PFFNet Model ]
"""

import argparse
import os

import torch
import utils
import torchvision
from PIL import Image
import torch.nn as nn
from torchvision import transforms

from model.rpnet import Net
from psnr_ssim import val as validate

device = utils.selectDevice()
mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
std  = torch.Tensor([0.229, 0.224, 0.225]).to(device)

def predict(opt):
    if not os.path.exists(opt.checkpoint):
        raise FileNotFoundError("File doesn't exists: {}".format(opt.checkpoint))

    net = utils.loadModel(opt.checkpoint, Net(opt.rb), dataparallel=True).to(device)
    net.eval()
        
    # Test photos: Default Reside
    images = utils.load_all_image(opt.hazy)

    # Output photos path
    os.makedirs(opt.dehazy, exist_ok=True)

    # -------------------------------------------------------------- #
    # Set the transform parameter,                                   #
    #   if normalize: Add the normalize params of the pretrain model #
    # -------------------------------------------------------------- #
    if opt.normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.ToTensor()

    for im_path in images:
        filename = im_path.split('/')[-1]
        im = Image.open(im_path)
        h, w = im.size
        
        print("==========> Input filename: {}".format(filename))
        
        im = transform(im).view(1, -1, w, h).to(device)
        
        with torch.no_grad():
            im_dehaze = net(im)

            if opt.normalize:
                im_dehaze = im_dehaze * std[:, None, None] + mean[:, None, None]
        
        # Take the Image out from GPU.
        im_dehaze = torch.clamp(im_dehaze, 0., 1.).cpu().data[0]
        im_dehaze = transforms.ToPILImage()(im_dehaze)

        im_dehaze.save(os.path.join(opt.dehazy, filename))
        print("==========> File saved: {}".format(os.path.join(opt.dehazy, filename)))

    return

def main(opt):
    # ----------------------- # 
    # Argument Error Handling #
    # ----------------------- #
    if not os.path.exists(opt.checkpoint):
        raise FileNotFoundError("File doesn't exists: {}".format(opt.checkpoint))

    if opt.cuda and torch.cuda.is_available():
        device = utils.selectDevice()

    tag = os.path.basename(os.path.dirname(opt.checkpoint))
    checkpoint = os.path.basename(opt.checkpoint).split('.')[0]
    opt.dehazy = os.path.join(opt.dehazy, tag, checkpoint)

    utils.details(opt, None)

    # Generate the images
    predict(opt)

    # Inference the performance on the validation set
    pass

    # Inference the performance on the test set
    gts = sorted(utils.load_all_image(opt.gt))
    dehazes = sorted(utils.load_all_image(opt.dehazy))
    validate(dehazes, gts, opt.dehazy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
    parser.add_argument("--rb", default=18, type=int, help="number of residual blocks")
    parser.add_argument("--checkpoint", type=str, help="root of model checkpoint")
    parser.add_argument("--hazy", default="./dataset/NTIRE2018_TEST/Hazy", type=str, help="path to load test images")
    parser.add_argument("--cuda", default=False, action='store_true', help="Use cuda?")
    # parser.add_argument("--gpus", default=8, type=int, help="nums of gpu to use")
    parser.add_argument("--dehazy", default="./output", type=str, help="path to save output images")
    # parser.add_argument("--record", default="record.txt", type=str, help="wrote the result to the textfile")
    parser.add_argument("--gt", default="./dataset/NTIRE2018_TEST/GT", type=str, help="path to load gt images")
    parser.add_argument("--normalize", default=False, action="store_true", help="pre / post normalization of the images")
    # parser.add_argument("--activation", default="LeakyReLU", help="the activation of the model")

    opt = parser.parse_args()
    
    # Main process
    os.system("clear")
    main(opt)
