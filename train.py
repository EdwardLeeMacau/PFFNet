"""
  FileName     [ train.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Train the model ]
"""

import argparse
import logging
import logging.config
import os
import pdb

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomCrop, Resize, ToTensor)
# from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

import utils
from data import DatasetFromFolder
from model.rpnet import Net

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--tag", type=str, help="tag for this training")
parser.add_argument("--rb", type=int, default=18, help="number of residual blocks")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1000, help="step to test the model performance. Default=2000")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--resume", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--pretrained", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--train", default="/media/disk1/IndoorTrain", type=str, help="path to load train datasets")
parser.add_argument("--test", default="/media/disk1/IndoorTest", type=str, help="path to load test datasets")
# subparser = parser.add_subparsers(required=True, dest="command", help="I-Haze / O-Haze / Both")

# ihazeparser = subparser.add_parser("I-Haze")
# ihazeparser.add_argument("--train", default="./IndoorTrain", type=str, help="path to load train datasets")
# ihazeparser.add_argument("--test", default="./IndoorTest", type=str, help="path to load test datasets")

# ohazeparser = subparser.add_parser("O-Haze")
# ohazeparser.add_argument("--train", default="./OutdoorTrain", type=str, help="path to load train datasets")
# ohazeparser.add_argument("--test", default="./OutdoorTest", type=str, help="path to load test datasets")

# Set logger
logging.config.fileConfig("logging.ini")
statelogger = logging.getLogger(__name__)

# Select Device
device = utils.selectDevice()

def main():
    global opt, name, model, criterion
    opt = parser.parse_args()
    print(opt)

    train_loss  = np.empty(0, dtype=float)
    psnr_epochs = np.empty((0, 5), dtype=float)
    ssim_epochs = np.empty((0, 5), dtype=float)
    mse_epochs  = np.empty((0, 5), dtype=float)
    epochs = np.empty(0, dtype=np.int64)
    
    # Tag_ResidualBlocks_BatchSize
    # name = "{}_{}_{}".format(opt.command, opt.rb, opt.batchSize)
    name = "{}_{}_{}".format("Indoor", opt.rb, opt.batchSize)

    # logger = SummaryWriter("runs/" + name)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = 1334
    torch.manual_seed(seed)
    if opt.cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    print("==========> Loading datasets")

    train_dataset = DatasetFromFolder(opt.train, transform=Compose([
        ToTensor()
    ]))

    test_dataset = DatasetFromFolder(opt.test, transform=Compose([
        ToTensor()
    ]))

    train_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, num_workers=opt.threads, batch_size=1, pin_memory=True, shuffle=False)

    print("==========> Building model")
    model = Net(opt.rb)
    criterion = nn.MSELoss(size_average=True)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] // 2 + 1
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise Exception("=> no checkpoint found at '{}'".format(opt.resume))
            # print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['state_dict'].state_dict())
        else:
            raise Exception("=> no pretrained model found at '{}'".format(opt.pretrained))
            # print("=> no model found at '{}'".format(opt.pretrained))

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
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    print("==========> Pre-Testing")
    mses, psnrs, ssims = test(test_loader, 0)
    mse_epochs  = np.append(mse_epochs, np.expand_dims(mses, axis=0), axis=0)
    psnr_epochs = np.append(psnr_epochs, np.expand_dims(psnrs, axis=0), axis=0)
    ssim_epochs = np.append(ssim_epochs, np.expand_dims(ssims, axis=0), axis=0)
    epochs = np.append(epochs, np.array([0]), axis=0)

    print("==========> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # Adjust the learning
        scheduler.step()

        # Train, val
        loss = train(train_loader, test_loader, optimizer, epoch)
        mses, psnrs, ssims = test(test_loader, epoch)

        train_loss  = np.append(train_loss, np.array([loss]), axis=0)
        mse_epochs  = np.append(mse_epochs, np.expand_dims(mses, axis=0), axis=0)
        psnr_epochs = np.append(psnr_epochs, np.expand_dims(psnrs, axis=0), axis=0)
        ssim_epochs = np.append(ssim_epochs, np.expand_dims(ssims, axis=0), axis=0)
        epochs = np.append(epochs, np.array([epoch]), axis=0)
        
        utils.save_checkpoint(model, "/media/disk1/checkpoints", epoch, name)

        with open("statistics.txt", "w") as textfile:
            textfile.write("Train Loss")
            textfile.write(str(train_loss.tolist()))

            textfile.write("\n")
            textfile.write("MSE")
            textfile.write(str(mse_epochs.tolist()))
            
            textfile.write("\n")
            textfile.write("PSNR")
            textfile.write(str(psnr_epochs.tolist()))
            
            textfile.write("\n")
            textfile.write("SSIM")
            textfile.write(str(ssim_epochs.tolist()))

            textfile.write("\n")
            textfile.write("Epochs")
            textfile.write(str(epochs.tolist()))

        # Plot TrainLoss and TestLoss
        plt.clf()
        plt.plot(epochs[1:], train_loss, label="TrainLoss vs Epochs", color='b')
        plt.plot(epochs, mse_epochs, label="TestLoss vs Epochs", color='r')
            
        plt.legend(loc=0)
        plt.title("Loss vs Epochs")
        plt.savefig("loss.png")

        # Plot PSNR and SSIM
        plt.clf()
        fig, axis1 = plt.subplots(sharex=True)
        axis1.set_xlabel('Epoch(s)')
        axis1.set_ylabel('Average PSNR')
        axis1.plot(epochs, psnr_epochs, label="PSNR vs Epochs", color='b')
        axis1.plot(epochs, np.repeat(np.amax(psnr_epochs), len(epochs)), ':')
        axis1.tick_params()
        
        axis2 = axis1.twinx()
        axis2.plot(epochs, ssim_epochs, label="SSIM vs Epochs", color='r')
        axis2.set_ylabel('Average SSIM')
        axis2.tick_params()
            
        plt.legend(loc=0)
        plt.title("PSNR-SSIM vs Epochs")
        plt.savefig("psnr_ssim.png")

def train(train_loader, test_loader, optimizer, epoch):
    statelogger.info("epoch: {}, lr: {}".format(epoch, optimizer.param_groups[0]["lr"]))

    trainLoss = []

    for iteration, batch in enumerate(train_loader, 1):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        steps = len(train_loader) * (epoch - 1) + iteration

        data, label = batch[0].to(device), batch[1].to(device)

        output = model(data)
        loss = criterion(output, label)
        loss.backward()

        trainLoss.append(loss.item())
        optimizer.step()

        if iteration % 10 == 0:
            statelogger.info("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(train_loader), loss.item()))
            # logger.add_scalar('loss', loss.data[0], steps)

        if iteration % opt.step == 0:
            data_temp = make_grid(data.data)
            label_temp = make_grid(label.data)
            output_temp = make_grid(output.data)

            torchvision.utils.save_image(data_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_data.png".format(epoch, iteration))
            torchvision.utils.save_image(label_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_label.png".format(epoch, iteration))
            torchvision.utils.save_image(output_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_output.png".format(epoch, iteration))

    trainLoss = np.asarray(trainLoss)
    return np.mean(trainLoss)

def test(test_loader, epoch):
    psnrs = []
    ssims = []
    mses = []
    model.eval()

    with torch.no_grad():
        for iteration, batch in enumerate(test_loader, 1):
            statelogger.info("Testing: {}".format(iteration))
            data   = batch[0].to(device)
            label  = batch[1].to(device)

            output = model(data)
            output = torch.clamp(output, 0., 1.)
            
            mse = nn.MSELoss()(output, label)
            mses.append(mse.item())
            
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            label  = label.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # psnr = 10 * np.log10(1.0 / mse.item())
            # psnrs.append(psnr)

            # Newly Added.
            psnr = compare_psnr(label, output)
            ssim = compare_ssim(label, output, multichannel=True)
            psnrs.append(psnr)
            ssims.append(ssim)
        
        psnr_mean = np.mean(psnrs)
        mse_mean  = np.mean(mses)
        ssim_mean = np.mean(ssims)

        statelogger.info("[Vaild] epoch: {}, mse: {}".format(epoch, mse_mean))
        statelogger.info("[Vaild] epoch: {}, psnr: {}".format(epoch, psnr_mean))
        statelogger.info("[Vaild] epoch: {}, ssim: {}".format(epoch, ssim_mean))

    return np.array(mses), np.array(psnrs), np.array(ssims)

if __name__ == "__main__":
    os.system('clear')
    main()
