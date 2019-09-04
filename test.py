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
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import psnr_ssim.val
from model.rpnet import Net
from psnr_ssim import val


device = 'cpu'
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
    makedirs = []
    folder = opt.dehazy
    while not os.path.exists(folder):
        makedirs.append(folder)
        folder = os.path.dirname(folder)
    
    while makedirs:
        makedirs, folder = makedirs[:-1], makedirs[-1]
        os.makedirs(folder, exist_ok=True)

    # --------------------------------------------------------------
    # Set the transform parameter,
    #   if normalize: Add the normalize params of the pretrain model
    # --------------------------------------------------------------
    if opt.normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.ToTensor()

    print("==========> DeHazing, Target: {}".format(len(images)))
    for im_path in tqdm(images):
        filename = im_path.split('/')[-1]
        im = Image.open(im_path)
        h, w = im.size
        
        print("==========> Input filename: {}".format(filename))
        
        im = transform(im).view(1, -1, w, h).to(device)
        
        with torch.no_grad():
            im_dehaze = net(im)

            # Shift the value to [0, 1]
            if opt.normalize:
                im_dehaze = im_dehaze * std[:, None, None] + mean[:, None, None]
        
        # Take the Image out from GPU.
        im_dehaze = torch.clamp(im_dehaze, 0., 1.).cpu().data[0]
        im_dehaze = transforms.ToPILImage()(im_dehaze)

        im_dehaze.save(os.path.join(opt.dehazy, filename))
        print("==========> File saved: {}".format(os.path.join(opt.dehazy, filename)))

    # Can add yield function

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
    parser.add_argument("--rb", default=18, type=int, help="number of residual blocks")
    parser.add_argument("--checkpoint", type=str, help="root of model checkpoint")
    parser.add_argument("--hazy", default="/media/disk1/EdwardLee/IndoorTest/hazy", type=str, help="path to load test images")
    parser.add_argument("--cuda", default=True, help="Use cuda?")
    parser.add_argument("--gpus", default=8, type=int, help="nums of gpu to use")
    parser.add_argument("--dehazy", default="/media/disk1/EdwardLee/Output", type=str, help="path to save output images")
    parser.add_argument("--record", default="./psnr_ssim.txt", type=str, help="wrote the result to the textfile")
    parser.add_argument("--gt", default="/media/disk1/EdwardLee/IndoorTest/gt", type=str, help="path to load gt images")
    parser.add_argument("--normalize", default=False, action="store_true", help="pre / post normalization of the images")
    parser.add_argument("--activation", default="LeakyReLU", help="the activation of the model")

    opt = parser.parse_args()

    tag = os.path.basename(os.path.dirname(opt.checkpoint))
    num_checkpoint = os.path.basename(opt.checkpoint).split('.')[0]
    opt.dehazy     = os.path.join(opt.dehazy, tag, num_checkpoint)

    utils.details(opt, None)

    # -------------------- #
    # Generate the images  #
    # -------------------- #
    predict(opt)

    # ---------------------------------------- #
    # Vaildate the performance on the test set #
    # ---------------------------------------- #
    gts = sorted(utils.load_all_image(opt.gt))
    dehazes = sorted(utils.load_all_image(opt.dehazy))
    psnr_ssim.val(dehazes, gts, opt.record)

if __name__ == "__main__":
    os.system("clear")
    main()
