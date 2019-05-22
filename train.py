"""
  FileName     [ train.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Train the model ]
"""

import argparse
import logging
import logging.config
import os
import pprint
from datetime import date

import numpy as np
import tensorboardX
import torch
import torchsummary
import torchvision
from matplotlib import pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomCrop, Resize, ToTensor)
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
from data import DatasetFromFolder
from model.rpnet import Net

parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
# Basic Training settings
parser.add_argument("--rb", type=int, default=12, help="number of residual blocks")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--activation", default="LeakyReLU", help="the activation function use at training")
parser.add_argument("--normalize", default=False, action="store_true", help="normalized the dataset images")
parser.add_argument("--milestones", type=int, nargs='*', default=[10], help="Which epoch to decay the learning rate")
parser.add_argument("--gamma", type=float, default=0.1, help="The ratio of decaying learning rate everytime")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--pretrained", type=str, help="path to pretrained model (default: none)")
# Message logging, model saving setting
parser.add_argument("--tag", type=str, default="Indoor_512", help="tag for this training")
parser.add_argument("--checkpoints", default="/media/disk1/EdwardLee/checkpoints", type=str, help="path to save the checkpoints")
parser.add_argument("--step", type=int, default=1000, help="step to test the model performance")
parser.add_argument("--log_interval", type=int, default=10, help="interval per iterations to log the message")
parser.add_argument("--save_interval", type=int, default=1, help="interval per epochs to save the model")
parser.add_argument("--detail", default="./train_details", help="the root directory to save the training details")
# Device setting
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_arugment("--fixrandomseed", default=False, help="train with fix random seed")
# Dataset loading, pretrain model setting
parser.add_argument("--resume", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--train", default="/media/disk1/EdwardLee/IndoorTrain_512", type=str, help="path to load train datasets")
parser.add_argument("--val", default="/media/disk1/EdwardLee/IndoorVal_512", type=str, help="path to load val datasets")

# subparser = parser.add_subparsers(required=True, dest="command", help="I-Haze / O-Haze")

# ihazeparser = subparser.add_parser("I-Haze")
# ihazeparser.add_argument("--train", default="/media/disk1/EdwardLee/IndoorTrain", type=str, help="path to load train datasets")
# ihazeparser.add_argument("--test", default="/media/disk1/EdwardLee/IndoorTest", type=str, help="path to load test datasets")

# ohazeparser = subparser.add_parser("O-Haze")
# ohazeparser.add_argument("--train", default="/media/disk1/EdwardLee/OutdoorTrain", type=str, help="path to load train datasets")
# ohazeparser.add_argument("--test", default="/media/disk1/EdwardLee/OutdoorTest", type=str, help="path to load test datasets")

opt = parser.parse_args()

# (Deprecated)
# Set logger
# logging.config.fileConfig("logging.ini")
# statelogger = logging.getLogger(__name__)

# Select Device
device = utils.selectDevice()
cudnn.benchmark = True

def main():
    global opt, name, model, criterion, writer

    # Establish the folder and the summarywriter
    name    = "{}_{}_{}_{}".format(opt.tag, date.today().strftime("%Y%m%d"), opt.rb, opt.batchSize)
    writer  = SummaryWriter("./{}/{}".format(opt.detail, name))
    
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if opt.fixrandomseed:
        seed = 1334
        torch.manual_seed(seed)
        
        if opt.cuda: 
            torch.cuda.manual_seed(seed)

    print("==========> Loading datasets")
    # -----------------------------------------------------------------
    # Normalization methods
    #   input[channel] = (input[channel] - mean[channel]) / std[channel]
    #
    # For pytorch pretrained ImageNet Feature Extractor
    #   mean = [0.485, 0.456, 0.406]
    #   std  = [0.229, 0.224, 0.225]
    # -----------------------------------------------------------------
    if opt.normalize:
        img_transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        img_transform = Compose([ToTensor()])
    
    train_dataset = DatasetFromFolder(opt.train, transform=img_transform)
    val_dataset   = DatasetFromFolder(opt.val, transform=img_transform)

    train_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True, shuffle=False)
    
    print("==========> Building model")
    model = Net(opt.rb)
    criterion = nn.MSELoss(size_average=True)
    
    # -------------------------
    # Perceptual Loss
    # -------------------------
    # vgg16 = ...

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise Exception("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['state_dict'].state_dict())
        else:
            raise Exception("=> no pretrained model found at '{}'".format(opt.pretrained))

    if opt.cuda:
        print("==========> Setting GPU")
        model = nn.DataParallel(model, device_ids=[i for i in range(opt.gpus)]).cuda()
        criterion = criterion.cuda()
    else:
        print("==========> Setting CPU")
        model = model.cpu()
        criterion = criterion.cpu()

    print("==========> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)

    # Extablish container
    loss_iter  = np.empty(0, dtype=float)
    psnr_iter  = np.empty(0, dtype=float)
    ssim_iter  = np.empty(0, dtype=float)
    mse_iter   = np.empty(0, dtype=float)
    iterations = np.empty(0, dtype=float)

    print("==========> Training setting")
    details(opt, "./{}/{}/{}".format(opt.detail, name, "args.txt"))

    print("==========> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step()

        loss_iter, mse_iter, psnr_iter, ssim_iter, iterations = train_eval(
            train_loader, val_loader, optimizer, epoch, loss_iter, mse_iter, psnr_iter, ssim_iter, iterations)
        
        # utils.save_checkpoint(model, opt.checkpoints, epoch, name)
        # mses, psnrs, ssims = test(val_loader, epoch, criterion)

    return

def train_eval(train_loader, val_loader, optimizer, epoch, loss_iter, mse_iter, psnr_iter, ssim_iter, iters):
    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
    
    trainLoss = []

    for iteration, (data, label) in enumerate(train_loader, 1):
        # ------------------
        # Train the network
        # ------------------
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        steps = len(train_loader) * (epoch - 1) + iteration

        data, label = data.to(device), label.to(device)

        output = model(data)
        loss = criterion(output, label)
        loss.backward()

        trainLoss.append(loss.item())
        optimizer.step()

        # -----------------------------------------------------
        # Log the training message, testing, saving the network
        # -----------------------------------------------------
        if steps % opt.log_interval == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(train_loader), loss.item()))
            writer.add_scalar('Train_loss', loss.data[0], steps)

        if steps % opt.save_interval == 0:
            data_temp   = make_grid(data.data, nrow=8)
            label_temp  = make_grid(label.data, nrow=8)
            output_temp = make_grid(output.data, nrow=8)

            torchvision.utils.save_image(data_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_data.png".format(epoch, iteration))
            torchvision.utils.save_image(label_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_label.png".format(epoch, iteration))
            torchvision.utils.save_image(output_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_output.png".format(epoch, iteration))

            # In epoch testing and saving (Newly added)
            utils.save_checkpoint(model, opt.checkpoints, epoch, name, iteration)        
            
        if steps % opt.step == 0:
            # mses, psnrs, ssims = test(val_loader, epoch, criterion)
            mse, psnr = test(val_loader, epoch, criterion)
            loss_iter = np.append(loss_iter, np.array([np.mean(trainLoss)]), axis=0)
            mse_iter  = np.append(mse_iter, np.array([mse]), axis=0)
            psnr_iter = np.append(psnr_iter, np.array([psnr]), axis=0)
            # ssim_iter = np.append(ssim_iter, np.array([ssim]), axis=0)
            iters     = np.append(iters, np.array([steps / len(train_loader)]), axis=0)

            trainLoss = []  # Clean the list 

            # (Deprecated)
            # mse_mean  = np.average(mse_iter, axis=1)
            # psnr_mean = np.average(psnr_iter, axis=1)
            # ssim_mean = np.average(ssim_iter, axis=1)
            
            with open(os.path.join(opt.detail, name, "statistics.txt"), "w") as textfile:
                datas = [str(data.tolist()) for data in (loss_iter, mse_iter, psnr_iter, ssim_iter, iters)]
                textfile.write("\n".join(datas))
                
            # ----------------------------------------------------------
            # Plot TrainLoss, TestLoss and the minimum value of TestLoss
            # ----------------------------------------------------------
            writer.add_scalar('Train_Loss', trainLoss, steps / len(train_loader))
            writer.add_scalar('Val_Loss', mse, steps / len(train_loader))
            writer.add_scalar('Val_PSNR', psnr, steps / len(train_loader))

    return loss_iter, mse_iter, psnr_iter, ssim_iter, iters

def details(opt, path):
    makedirs = []
    
    folder = os.path.dirname(path)
    while not os.path.exists(folder):
        makedirs.append(folder)
        folder = os.path.dirname(folder)

    while len(makedirs) > 0:
        makedirs, folder = makedirs[:-1], makedirs[-1]
        os.makedirs(folder)

    with open(path, "w") as textfile:
        for item, values in vars(opt).items():
            msg = "{:16} {}".format(item, values)
            print(msg)
            textfile.write(msg)

    # torchsummary.summary(model, (3, 512, 512), batch_size=16, device='cuda')

    return folder

def draw_graphs(train_loss, val_loss, psnr, ssim, x, iters_per_epoch, 
            loss_filename="loss.png", loss_log_filename="loss_log.png", psnr_filename="psnr.png"):
    # Linear scale of loss curve
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, train_loss, label="TrainLoss", color='b')
    plt.plot(x, val_loss, label="ValLoss", color='r')
    plt.plot(x, np.repeat(np.amin(val_loss), len(x)), ':')
    plt.legend(loc=0)
    plt.xlabel("Epoch(s) / Iteration: {}".format(iters_per_epoch))
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(opt.detail, name, loss_filename))

    # Log scale of loss curve
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, train_loss, label="TrainLoss", color='b')
    plt.plot(x, val_loss, label="ValLoss", color='r')
    plt.plot(x, np.repeat(np.amin(val_loss), len(x)), ':')
    plt.legend(loc=0)
    plt.xlabel("Epoch(s) / Iteration: {}".format(iters_per_epoch))
    plt.yscale('log')
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(opt.detail, name, loss_log_filename))

    # Linear scale of PSNR, SSIM
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    
    plt.plot(x, psnr, label="PSNR", color='b')
    plt.plot(x, np.repeat(np.amax(psnr), len(x)), ':')
    plt.xlabel("Epochs(s) / Iteration: {}".format(iters_per_epoch))
    plt.title("PSNR vs Epochs")

    # fig, axis1 = plt.subplots(sharex=True, figsize=(12.8, 7.2))
    # axis1.set_xlabel('Epoch(s) / Iteration: {}'.format(iters_per_epoch))
    # axis1.set_ylabel('Average PSNR')
    # axis1.plot(x, psnr, label="PSNR", color='b')
    # axis1.plot(x, np.repeat(np.amax(psnr), len(x)), ':')
    # axis1.tick_params()
    # axis2 = axis1.twinx()
    # axis2.plot(x, ssim, label="SSIM", color='r')
    # axis2.set_ylabel('Average SSIM')
    # axis2.tick_params()
        
    plt.legend(loc=0)
    # plt.title("PSNR-SSIM vs Epochs")
    plt.savefig(os.path.join(opt.detail, name, psnr_filename))

    return

def test(loader, epoch, criterion):
    """
      Params:
      - loader

      Return:
      - np.mean(mse)
      - np.mean(psnr)
    """
    psnrs = []
    ssims = []
    mses = []
    model.eval()

    with torch.no_grad():
        for (data, label) in tqdm(loader):
            batchsize = data.shape[0]

            data, label = data.to(device), label.to(device)

            # -----------------------------------------------------------------
            # Normalization methods
            #   input[channel] = (input[channel] - mean[channel]) / std[channel]
            #
            #   mean = [0.485, 0.456, 0.406]
            #   std  = [0.229, 0.224, 0.225]
            # 
            # Notes: 20190515
            #   The original model doesn't set any activation function in the output layer.
            # -----------------------------------------------------------------
            output = model(data)
            output = torch.clamp(output, 0., 1.)
            
            mse = criterion(output, label).item()
            mses.append(mse)
            
            # (batchsize, width, height, channel)
            output = output.permute(0, 2, 3, 1).cpu().numpy()
            label  = label.permute(0, 2, 3, 1).cpu().numpy()

            psnr = 10 * np.log10(1.0 / mse)
            psnrs.append(psnr)

            # for i in range(batchsize):
            #     psnr = compare_psnr(label[i], output[i])
            #     ssim = compare_ssim(label[i], output[i], multichannel=True)
            #     psnrs.append(psnr)
            #     ssims.append(ssim)
        
        psnr_mean = np.mean(psnrs)
        mse_mean  = np.mean(mses)
        # ssim_mean = np.mean(ssims)

        print("[Vaild] epoch: {}, mse: {}".format(epoch, mse_mean))
        print("[Vaild] epoch: {}, psnr: {}".format(epoch, psnr_mean))
        # print("[Vaild] epoch: {}, ssim: {}".format(epoch, ssim_mean))

    return np.mean(mses), np.mean(psnrs) #, np.mean(ssims)

if __name__ == "__main__":
    os.system('clear')    
    main()
