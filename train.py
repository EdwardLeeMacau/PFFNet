"""
  FileName     [ train.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Train the model ]
"""

import argparse
import os
from datetime import date

import matplotlib
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
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
from data import DatasetFromFolder
from model.rpnet import Net

# Select Device
device = utils.selectDevice()
cudnn.benchmark = True

mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
std  = torch.Tensor([0.229, 0.224, 0.225]).to(device)

# ----------------------------------------------------------------- #
# Normalization (mean shift)                                        #
# ----------------------------------------------------------------- #
# Normalization methods                                             #
#   input = (input - mean[:, None, None]) / std[:, None, None]      #
#                                                                   #
# Reverse Operation:                                                #
#   input = (input * std[:, None, None]) + mean[:, None, None]      #
#                                                                   #
# Source code:                                                      #
#   tensor.sub_(mean[:, None, None]).div_(std[:, None, None])       #
#                                                                   #
# Pretrain network normalize parameters                             #
#   mean = [0.485, 0.456, 0.406]                                    #
#   std  = [0.229, 0.224, 0.225]                                    #
# ----------------------------------------------------------------- #

def main(opt):
    if opt.fixrandomseed:
        seed = 1334
        torch.manual_seed(seed)
        
        if opt.cuda: 
            torch.cuda.manual_seed(seed)

    print("==========> Loading datasets")
    if opt.normalize:
        img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("==========> Using Normalization")
    else:
        img_transform = ToTensor()

    train_dataset = DatasetFromFolder(opt.train, transform=img_transform)
    val_dataset   = DatasetFromFolder(opt.val, transform=img_transform)
    train_loader  = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchsize, pin_memory=True, shuffle=True)
    val_loader    = DataLoader(dataset=val_dataset, num_workers=opt.threads, batch_size=opt.batchsize, pin_memory=True, shuffle=True)
    
    print("==========> Building model")
    # ------------------------------------------------------------------------------ #
    # Notes: 20190515                                                                #
    #   The original model doesn't set any activation function in the output layer.  #
    # ------------------------------------------------------------------------------ #
    model = Net(opt.rb)
    
    # ------------------------------------------- #
    # Loss: L1 Norm / L2 Norm / Perceptual loss   #
    # ------------------------------------------- #
    criterion = nn.MSELoss(size_average=True)
    
    # vgg16 = ...

    # --------------------------------------- #
    # Optimizer and learning rate scheduler   #
    # --------------------------------------- #
    print("==========> Setting Optimizer: {}".format(opt.optimizer))
    if opt.optimizer == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == "ASGD":
        optimizer = optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, lambd=1e-4, alpha=0.75, t0=1000000.0, weight_decay=opt.weight_decay)
    elif opt.optimizer == "Adadelta":
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, rho=0.9, eps=1e-06, weight_decay=opt.weight_decay)
    elif opt.optimizer == "Adagrad":
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, lr_decay=0, weight_decay=opt.weight_decay, initial_accumulator_value=0)
    elif opt.optimizer == "SparseAdam":
        optimizer = optim.SparseAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-08)
    elif opt.optimizer == "Adamax":
        optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-08, weight_decay=opt.weight_dacay)
    else:
        raise argparse.ArgumentError

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)

    # Option: resume training process from checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            model, optimizer, opt.starts, opt.iterations, scheduler = utils.loadCheckpoint(opt.resume, model, optimizer, scheduler)
        else:
            raise Exception("=> no checkpoint found at '{}'".format(opt.resume))

    # Option: load the weights from a pretrain network
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading pretrained model '{}'".format(opt.pretrained))
            model = utils.loadModel(opt.pretrained, model)
        else:
            raise Exception("=> no pretrained model found at '{}'".format(opt.pretrained))

    # Select training device
    if opt.cuda:
        print("==========> Setting GPU")
        model = nn.DataParallel(model, device_ids=[i for i in range(opt.gpus)]).cuda()
        criterion = criterion.cuda()
    else:
        print("==========> Setting CPU")
        model = model.cpu()
        criterion = criterion.cpu()

    # Create container
    loss_iter  = np.empty(0, dtype=float)
    psnr_iter  = np.empty(0, dtype=float)
    ssim_iter  = np.empty(0, dtype=float)
    mse_iter   = np.empty(0, dtype=float)
    iterations = np.empty(0, dtype=float)

    # Show training settings 
    print("==========> Training setting")
    utils.details(opt, "./{}/{}/{}".format(opt.detail, name, "args.txt"))

    # Set plotter to plot the loss curves
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(19.2, 10.8))

    # Start training
    print("==========> Training")
    for epoch in range(opt.starts, opt.epochs + 1):
        scheduler.step()

        # Main process
        loss_iter, mse_iter, psnr_iter, ssim_iter, iterations = train_val(
            model, optimizer, criterion, train_loader, val_loader, scheduler, 
            epoch, loss_iter, mse_iter, psnr_iter, ssim_iter, iterations, opt, name, fig, ax
        )

    return

def train_val(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                   scheduler: optim.lr_scheduler.MultiStepLR, epoch: int, loss_iter, mse_iter, psnr_iter, ssim_iter, iters, opt, name, fig, ax):
    print("===> lr: ", optimizer.param_groups[0]["lr"])
    
    trainLoss = []

    for iteration, (data, label) in enumerate(train_loader, 1):
        # Train the network 
        model.train()
        optimizer.zero_grad()

        steps = len(train_loader) * (epoch - 1) + iteration

        data, label = data.to(device), label.to(device)

        output = model(data)
        loss = criterion(output, label)
        loss.backward()

        trainLoss.append(loss.item())
        optimizer.step()

        # ----------------------------------------------------- #
        # 1. Log the training message                           #
        # 2. Plot the gradient of each layer (Deprecated)       #
        # 3. Validate the model                                 #
        # 4. Saving the network                                 #
        # ----------------------------------------------------- #
        # 1. Log the training message
        if steps % opt.log_interval == 0:
            print("===> [Epoch {}] [{:4d}/{:4d}]: Loss: {:.6f}".format(epoch, iteration, len(train_loader), loss.item()))
        
        # 2. Plot the gradient of each layer 
        # (Deprecated)

        # 3. Validate the model
        if steps % opt.save_interval == 0:
            # In epoch testing and saving
            checkpoint_path = os.path.join(opt.checkpoints, name, "{}_{}.pth".format(epoch, iteration))
            utils.saveCheckpoint(checkpoint_path, model, optimizer, scheduler, epoch, iteration)
        
        # 4. Saving the network
        if steps % opt.val_interval == 0:
            # mses, psnrs, ssims = test(val_loader, epoch, criterion)

            mse, psnr = validate(model, val_loader, criterion, epoch, iteration, normalize=opt.normalize)
            loss_iter = np.append(loss_iter, np.array([np.mean(trainLoss)]), axis=0)
            mse_iter  = np.append(mse_iter, np.array([mse]), axis=0)
            psnr_iter = np.append(psnr_iter, np.array([psnr]), axis=0)
            iters     = np.append(iters, np.array([steps / len(train_loader)]), axis=0)

            # Clean up the list
            trainLoss = []

            # Record the loss
            with open(os.path.join(opt.detail, name, "statistics.txt"), "w") as textfile:
                datas = [ str(data.tolist()) for data in (loss_iter, mse_iter, psnr_iter, ssim_iter, iters) ]
                textfile.write("\n".join(datas))
                
            # Plot TrainLoss, valloss
            draw_graphs(loss_iter, mse_iter, psnr_iter, ssim_iter, iters, len(train_loader), os.path.join(opt.detail, name), fig, ax)

    return loss_iter, mse_iter, psnr_iter, ssim_iter, iters

def draw_graphs(train_loss, val_loss, psnr, ssim, x, iters_per_epoch, savepath, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, loss_filename="loss.png"):
    """
    Plot out learning rate, training loss, validation loss and PSNR.

    Parameters
    ----------
    train_loss, val_loss, psnr, ssim, x

    iters_per_epoch

    savepath

    fig, ax

    loss_filename, loss_log_filename, psnr_filename : str
    """
    # Linear scale of loss curve
    ax[0].plot(x, train_loss, label="TrainLoss", color='b')
    ax[0].plot(x, val_loss, label="ValLoss", color='r')
    ax[0].plot(x, np.repeat(np.amin(val_loss), len(x)), ':')
    ax[0].legend(loc=0)
    ax[0].xlabel("Epoch(s) / Iteration: {}".format(iters_per_epoch))
    ax[0].title("Loss vs Epochs")
    
    # Log scale of loss curve
    ax[1].plot(x, train_loss, label="TrainLoss", color='b')
    ax[1].plot(x, val_loss, label="ValLoss", color='r')
    ax[1].plot(x, np.repeat(np.amin(val_loss), len(x)), ':')
    ax[1].legend(loc=0)
    ax[1].xlabel("Epoch(s) / Iteration: {}".format(iters_per_epoch))
    ax[1].yscale('log')
    ax[1].title("Loss vs Epochs")

    # Linear scale of PSNR, SSIM
    ax[2].plot(x, psnr, label="PSNR", color='b')
    ax[2].plot(x, np.repeat(np.amax(psnr), len(x)), ':')
    ax[2].xlabel("Epochs(s) / Iteration: {}".format(iters_per_epoch))
    ax[2].title("PSNR vs Epochs")

    plt.savefig(loss_filename)

    return fig, ax

def grid_show(model: nn.Module, loader: DataLoader, folder, nrow=8, normalize=False):
    """ Moved to graphs.py """
    iterator    = iter(loader)
    data, label = next(iterator)
    output      = model(data)

    if normalize:
        data   = data * std[:, None, None] + mean[:, None, None]
        label  = label * std[:, None, None] + mean[:, None, None]
        output = output * std[:, None, None] + mean[:, None, None]

    images = torch.cat((data, output, label), axis=0)
    images = make_grid(images.data, nrow=nrow)

    torchvision.utils.save_image(images, "{}/{}_{}.png".format(epoch, iteration))

    return

def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, epoch, iteration, normalize=False):
    """
    Validate the model

    Parameters
    ----------
    model : nn.Module
        The neural networks to train

    loader : torch.utils.data.DataLoader
        The training data
    
    epoch :
    
    criterion : 
    
    normalize

    Return
    ------
    mse : np.float
      - np.mean(mse)
    
    psnr : np.float
      - np.mean(psnr)
    """
    psnrs, mses = [], []
    model.eval()

    if normalize:
        print("==========> Using Normalization to measure MSE.")

    with torch.no_grad():
        for (data, label) in tqdm(loader):
            data, label = data.to(device), label.to(device)

            output = model(data)
            mse = criterion(output, label).item()
            mses.append(mse)

            if normalize:
                data   = data * std[:, None, None] + mean[:, None, None]
                label  = label * std[:, None, None] + mean[:, None, None]
                output = output * std[:, None, None] + mean[:, None, None]

            mse  = criterion(output, label).item()
            psnr = 10 * np.log10(1.0 / mse)
            psnrs.append(psnr)

        # grid_show(model, loader, os.path.join(opt.log, name))
        print("[Vaild] epoch: {}, mse:  {}".format(epoch, np.mean(mse)))
        print("[Vaild] epoch: {}, psnr: {}".format(epoch, np.mean(psnr)))

    return np.mean(mses), np.mean(psnrs)

if __name__ == "__main__":
    # Clean up OS screen
    os.system('clear')

    parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")

    # Basic Training settings
    parser.add_argument("--rb", default=18, type=int, help="number of residual blocks")
    parser.add_argument("--batchsize", default=16, type=int, help="training batch size")
    parser.add_argument("--epochs", default=15, type=int, help="number of epochs to train for")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning Rate. Default=1e-4")
    # parser.add_argument("--activation", default="LeakyReLU", type=str, help="the activation function use at training")
    parser.add_argument("--normalize", default=True, action="store_true", help="normalized the dataset images")
    parser.add_argument("--milestones", default=[10], type=int, nargs='*', help="Which epoch to decay the learning rate")
    parser.add_argument("--gamma", default=0.1, type=float, help="The ratio of decaying learning rate everytime")
    parser.add_argument("--starts", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
    parser.add_argument("--pretrained", type=str, help="path to pretrained model (default: none)")
    parser.add_argument("--weight_decay", default=0, type=float, help="The weight penalty in the training")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Choose the optimizer")

    # Message logging, model saving setting
    parser.add_argument("--tag", default="Indoor", type=str, help="tag for this training")
    parser.add_argument("--checkpoints", default="./checkpoints", type=str, help="Path to save checkpoints")
    parser.add_argument("--val_interval", default=1000, type=int,  help="step to test the model performance")
    parser.add_argument("--log_interval", default=10, type=int, help="interval per iterations to log the message")
    # parser.add_argument("--grad_interval", default=0, type=int, help="interval per iterations to draw the gradient")
    parser.add_argument("--save_interval", default=1000, type=int, help="interval per iterations to save the model")
    parser.add_argument("--detail", default="./log", type=str, help="the root directory to save the training details")

    # Device setting
    parser.add_argument("--cuda", default=True, type=bool, help="Use cuda?")
    parser.add_argument("--gpus", default=1, type=int, help="nums of gpu to use")
    parser.add_argument("--threads", default=8, type=int, help="Number of threads for data loader to use.")
    parser.add_argument("--fixrandomseed", default=False, help="train with fix random seed")

    # Pretrain model setting
    parser.add_argument("--resume", type=str, help="Path to checkpoint.")

    # Dataloader setting
    parser.add_argument("--train", default="./dataset/ntire2018_640", type=str, help="path to load train datasets")
    parser.add_argument("--val", default="./dataset/ntire2018_val", type=str, help="path to load val datasets")

    opt = parser.parse_args()

    # Check arguments
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    if opt.resume and opt.pretrained:
        raise ValueError("opt.resume and opt.pretrain should not be True in the same time.")

    if opt.resume and (not os.path.isfile(opt.resume)):
        raise ValueError("{} doesn't not exists".format(opt.resume))

    if opt.pretrained and (not os.path.isfile(opt.pretrained)):
        raise ValueError("{} doesn't not exists".format(opt.pretrained))

    # Check data directory
    for path in (opt.train, opt.val):
        if not os.path.exists(path):
            raise ValueError("{} doesn't exist".format(path))

    # Make file directories
    name = "{}_{}_{}_{}".format(opt.tag, date.today().strftime("%Y%m%d"), opt.rb, opt.batchsize)
    os.makedirs(os.path.join(opt.checkpoints, name), exist_ok=True)

    # Execute main process
    main(opt)
