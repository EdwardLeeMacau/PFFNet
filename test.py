#!/usr/bin/env python
"""
  FileName     [ test.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Generated the dehazed images by PFFNet Model ]

  Usage:
  1. Use GPU to inference dehazy image
  >>> python test.py --checkpoint <checkpoint_dir> --cuda

  2. Use CPU to inference dehazy image
  >>> python test,py --checkpoint <checkpoint_dir>

  3. If the model is need to normalize the image before
  >>> python test.py --checkpoint <checkpoint_dir> --normalize
"""

import argparse
import os

import torch
import utils
import torchvision
from PIL import Image
import torch.nn as nn
from torchvision import transforms

import graphs
from model.net import MeanShift, InverseMeanShift
from model.rpnet import Net
from model.rpnet_improve import ImproveNet
from validate import val as validate

# device = utils.selectDevice()
# mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
# std  = torch.Tensor([0.229, 0.224, 0.225]).to(device)

def predict(opt, net, folder, output_folder, device, *args):
    """ 
    Generate the result. 

    Parameters
    ----------
    folder, output : str
        Input directory and output directory

    net : nn.Module
        (...)

    normalized : bool
        (...)
    """
    # net = utils.loadModel(opt.checkpoint, ImproveNet(opt.rb), dataparallel=True).to(device)
    # net.eval()
        
    # Test photos: Default Reside
    images = utils.load_all_image(opt.hazy)

    if opt.normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.ToTensor()

    # Inference
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
        im = im.cpu()
        im_dehaze = torch.clamp(im_dehaze, 0., 1.).cpu().data[0]
        im_dehaze = transforms.ToPILImage()(im_dehaze)

        im_dehaze.save(os.path.join(opt.dehazy, filename))
        print("==========> File saved: {}".format(os.path.join(opt.dehazy, filename)))

    return

def main(opt):
    """ Main process of test.py """
    # Load Model
    net = utils.loadModel(opt.checkpoint, ImproveNet(opt.rb)), dataparallel=True)

    if opt.cuda:
        device = 'gpu'

    if not opt.cuda:
        device = 'cpu'

    if opt.normalize:
        net = nn.Sequential(
            net, 
            InverseMeanShift(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ).to(device)

    # Inference the images on the test set
    predict(opt, net, opt.hazy, opt.dehazy, device)

    # Measure the performance on the validation set
    # predict(validate)

    # Measure the performance on the test set
    gts     = sorted(utils.load_all_image(opt.gt))
    dehazes = sorted(utils.load_all_image(opt.dehazy))

    if opt.record is None:
        validate(dehazes, gts)

    if opt.record is not None:
        validate(dehazes, gts, os.path.join(opt.dehazy, opt.record))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
    parser.add_argument("--rb", default=18, type=int, help="number of residual blocks")
    parser.add_argument("--checkpoint", type=str, required=True, help="root of model checkpoint")
    parser.add_argument("--hazy", default="./dataset/NTIRE2018_TEST/Hazy", type=str, help="path to load hazy images")
    parser.add_argument("--cuda", default=False, action='store_true', help="Use cuda?")
    parser.add_argument("--dehazy", default="./output", type=str, help="path to save output images")
    parser.add_argument("--record", default="inference.xlsx", type=str, help="Write result to spreadsheet")
    parser.add_argument("--gt", default="./dataset/NTIRE2018_TEST/GT", type=str, help="path to load gt images")
    parser.add_argument("--normalize", default=False, action="store_true", help="pre / post normalization of the images")

    opt = parser.parse_args()

    # Modify the opt
    tag = os.path.basename(os.path.dirname(opt.checkpoint))
    checkpoint = os.path.basename(opt.checkpoint).split('.')[0]
    opt.dehazy = os.path.join(opt.dehazy, tag, checkpoint)

    # Argument Error Handling
    if not os.path.exists(opt.checkpoint):
        raise FileNotFoundError("File doesn't exists: {}".format(opt.checkpoint))

    if not os.path.exists(opt.dehazy):
        os.makedirs(opt.dehazy)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # Main process
    os.system("clear")
    main(opt)
