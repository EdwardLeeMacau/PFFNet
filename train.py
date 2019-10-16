"""
  FileName     [ train.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Train the model ]

  Usage:
  >>> python train.py --normalized --cuda
"""

import argparse
import os
import shutil
from datetime import date

import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib import gridspec
from skimage.measure import compare_psnr, compare_ssim
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomCrop, Resize, ToTensor)
from torchvision.utils import make_grid

import cmdparser
import graphs
import utils
from model import lossnet
from data import DatasetFromFolder
from model.rpnet import Net
from model.rpnet_improve import ImproveNet
from model.lossnet import LossNetwork

# Select Device
device = utils.selectDevice()
cudnn.benchmark = True

# Normalization(Mean Shift)
mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
std  = torch.Tensor([0.229, 0.224, 0.225]).to(device)

def getDataset(opt, transform):
    """ 
    Return the dataloader object 

    Parameters
    ----------
    opt : namespace

    transform : torchvision.transform

    Return
    ------
    train_loader, val_loader : torch.utils.data.DataLoader
    """
    train_dataset = DatasetFromFolder(opt.train, transform=transform)
    val_dataset   = DatasetFromFolder(opt.val, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset, 
        num_workers=opt.threads, 
        batch_size=opt.batchsize, 
        pin_memory=True, 
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset, 
        num_workers=opt.threads, 
        batch_size=opt.batchsize, 
        pin_memory=True, 
        shuffle=True
    )

    return train_loader, val_loader

def getOptimizer(model, opt):
    """ 
    Return the optimizer (and schedular)

    Parameters
    ----------
    opt : namespace

    Return
    ------
    optimizer : torch.optim
    """
    if opt.optimizer == "Adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "ASGD":
        optimizer = optim.ASGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            lambd=1e-4,
            alpha=0.75,
            t0=1000000.0,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "Adadelta":
        optimizer = optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            rho=0.9,
            eps=1e-06,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "Adagrad":
        optimizer = optim.Adagrad(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            lr_decay=0,
            weight_decay=opt.weight_decay,
            initial_accumulator_value=0
        )
    elif opt.optimizer == "Adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "ASGD":
        optimizer = optim.ASGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            lambd=1e-4,
            alpha=0.75,
            t0=1000000.0,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "Adadelta":
        optimizer = optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            rho=0.9,
            eps=1e-06,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "Adagrad":
        optimizer = optim.Adagrad(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            lr_decay=0,
            weight_decay=opt.weight_decay,
            initial_accumulator_value=0
        )
    elif opt.optimizer == "SparseAdam":
        optimizer = optim.SparseAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            betas=(opt.b1, opt.b2),
            eps=1e-08
        )
    elif opt.optimizer == "Adamax":
        optimizer = optim.Adamax(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            betas=(opt.b1, opt.b2),
            eps=1e-08,
            weight_decay=opt.weight_dacay
        )
    else:
        raise ValueError(opt.optimizer, " doesn't exist.")

    return optimizer

def getFigureSpec(iteration: int, perceptual: bool):
    """
    Get 2x2 Figure And Axis

    Return
    ------
    fig, axis : matplotlib.figure.Figure, matplotlib.axes.Axes
        The plotting instance.
    """
    fig, grids = plt.figure(figsize=(19.2, 10.8)), gridspec.GridSpec(2, 2)

    axis = [ fig.add_subplot(gs) for gs in grids ] 
    for ax in axis:
        ax.set_xlabel("Epoch(s) / Iteration: {}".format(iteration))

    # Linear scale of Loss
    axis[0].set_ylabel("Image Loss")
    axis[0].set_title("Loss")

    # Log scale of Loss
    axis[1].set_yscale("log")
    axis[1].set_ylabel("Image Loss")
    axis[1].set_title("Loss (Log scale)")

    # PSNR
    axis[2].set_title("Average PSNR")

    # Learning Rate 
    axis[3].set_yscale('log')
    axis[3].set_title("Learning Rate")

    # Add TwinScale for Perceptual Loss
    if perceptual:
        axis.append( axis[0].twinx() )
        axis[4].set_ylabel("Perceptual Loss")
        axis.append( axis[1].twinx() )
        axis[5].set_ylabel("Perceptual Loss")

    return fig, axis

def getPerceptualModel(model):
    """ 
    Return the Perceptual Model

    Return
    ------
    perceptual : {nn.Module, None}
    """
    perceptual = None

    if opt.perceptual == 'vgg16':
        print("==========> Using VGG16 as Perceptual Loss Model")
        perceptual = LossNetwork(
            torchvision.models.vgg16(pretrained=True),
            lossnet.VGG16_Layer
        )

    if opt.perceptual == 'vgg16_bn':
        print("==========> Using VGG16 with Batch Normalization as Perceptual Loss Model")
        perceptual = LossNetwork(
            torchvision.models.vgg16_bn(pretrained=True),
            lossnet.VGG16_bn_Layer
        )

    if opt.perceptual == 'vgg19':
        print("==========> Using VGG19 as Perceptual Loss Model")
        perceptual = LossNetwork(
            torchvision.models.vgg19(pretrained=True),
            lossnet.VGG19_Layer
        )

    if opt.perceptual == 'vgg19_bn':
        print("==========> Using VGG19 with Batch Normalization as Perceptual Loss Model")
        perceptual = LossNetwork(
            torchvision.models.vgg19_bn(pretrained=True),
            lossnet.VGG19_bn_Layer
        )

    if opt.perceptual == "resnet18":
        print("==========> Using Resnet18 as Perceptual Loss Model")
        perceptual = LossNetwork(
            torchvision.models.resnet18(pretrained=True),
            lossnet.Resnet18_Layer
        )

    if opt.perceptual == "resnet34":
        print("==========> Using Resnet34 as Perceptual Loss Model")
        perceptual = LossNetwork(
            torchvision.models.resnet34(pretrained=True),
            lossnet.Resnet34_Layer
        )

    if opt.perceptual == "resnet50":
        print("==========> Using Resnet50 as Perceptual Loss Model")
        perceptual = LossNetwork(
            torchvision.models.resnet50(pertrained=True),
            lossnet.Resnet50_Layer
        )

    return perceptual

def main(opt):
    """ 
    Main process of train.py 

    Parameters
    ----------
    opt : namespace
        The option (hyperparameters) of these model
    """
    if opt.fixrandomseed:
        seed = 1334
        torch.manual_seed(seed)
        
        if opt.cuda: 
            torch.cuda.manual_seed(seed)

    print("==========> Loading datasets")
    # TODO: Normalize Layer Handling
    if opt.normalize:
        img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        img_transform = ToTensor()

    # Dataset 
    train_loader, val_loader = getDataset(opt, img_transform)

    # TODO
    # Load Model
    print("==========> Building model")
    model = ImproveNet(opt.rb)
    
    # --------------------------------- #
    # Loss: L1 Norm / L2 Norm           #
    #   Perceptual Model (Optional)     # 
    #   TODO Append Layer (Optional)    #
    # --------------------------------- #
    criterion  = nn.MSELoss(reduction='mean')
    perceptual = None
    if not opt.perceptual is None:
        perceptual = getPerceptualModel(opt.perceptual).eval()

    # --------------------------------------- #
    # Optimizer and learning rate scheduler   #
    # --------------------------------------- #
    print("==========> Setting Optimizer: {}".format(opt.optimizer))
    optimizer = getOptimizer(model, opt)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)

    # ----------------------------------------------- #
    # Option: resume training process from checkpoint #
    # ----------------------------------------------- #
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            model, optimizer, opt.starts, opt.iterations, scheduler = utils.loadCheckpoint(opt.resume, model, optimizer, scheduler)
        else:
            raise Exception("=> no checkpoint found at '{}'".format(opt.resume))

    # ------------------------------------------------ #
    # Option: load the weights from a pretrain network #
    # ------------------------------------------------ #
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading pretrained model '{}'".format(opt.pretrained))
            model = utils.loadModel(opt.pretrained, model)
        else:
            raise Exception("=> no pretrained model found at '{}'".format(opt.pretrained))

    # -------------------------------------------------- #
    # Select training device                             #
    #   if GPU is available and --cuda -> train with GPU #
    #   if not --cuda, train with CPU                    #
    # -------------------------------------------------- #
    if opt.cuda:
        print("==========> Setting GPU")

        model = nn.DataParallel(model, device_ids=[i for i in range(opt.gpus)]).cuda()
        criterion = criterion.cuda()

        if perceptual is not None:
            perceptual = perceptual.cuda()
    else:
        print("==========> Setting CPU")
        
        model = model.cpu()
        criterion = criterion.cpu()

        if perceptual is not None:
            perceptual = perceptual.cpu()

    # Create container
    loss_iter  = np.empty(0, dtype=float)
    perc_iter  = np.empty(0, dtype=float)
    psnr_iter  = np.empty(0, dtype=float)
    ssim_iter  = np.empty(0, dtype=float)
    mse_iter   = np.empty(0, dtype=float)
    lr_iter    = np.empty(0, dtype=float)
    iterations = np.empty(0, dtype=float)

    # -------------------------------------- #
    # Set plotter to plot the loss curves    #
    # -------------------------------------- #
    twinx = (opt.perceptual is not None)
    fig, axis = getFigureSpec(len(train_loader), twinx)

    # ------------------------ #
    # Start training           #
    # ------------------------ #
    print("==========> Training")
    for epoch in range(opt.starts, opt.epochs + 1):
        loss_iter, perc_iter, mse_iter, psnr_iter, ssim_iter, lr_iter, iterations, _, _ = train_val(
            model, optimizer, criterion, perceptual, train_loader, val_loader, scheduler, 
            epoch, loss_iter, perc_iter, mse_iter, psnr_iter, ssim_iter, lr_iter, iterations, opt, name, fig, axis
        )

        scheduler.step()

    return

def train_val(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, perceptual: LossNetwork, train_loader: DataLoader, val_loader: DataLoader, scheduler: optim.lr_scheduler.MultiStepLR, epoch: int, loss_iter, perc_iter, mse_iter, psnr_iter, ssim_iter, lr_iter, iters, opt, name, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes):
    """
    Main function of training and vaildation

    Parameters
    ----------
    model, optimizer, criterion : nn.Module, optim.Optimizer, nn.Module
        The main elements of the Neural Network

    perceptual : {nn.Module, None} optional
        Pass None or a pretrained Neural Network to calculate perceptual loss

    train_loader, val_loader : DataLoader
        The training and validation dataset

    scheduler : optim.lr_scheduler.MultiStepLR
        Learning rate scheduler

    epoch : int
        The processing train epoch

    loss_iter, perc_iter, mse_iter, psnr_iter, ssim_iter, iters : 1D-Array like
        The container to record the training performance

    opt : namespace
        The training option

    name : str
        (...)
    
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        (...)
    """
    trainloss = []
    perceloss = []

    for iteration, (data, label) in enumerate(train_loader, 1):
        model.train()
        optimizer.zero_grad()

        steps = len(train_loader) * (epoch - 1) + iteration

        data, label = data.to(device), label.to(device)
        output = model(data)

        # Calculate loss
        image_loss = criterion(output, label)

        if perceptual is not None:
            perceptual_loss = perceptual(output, label)

        # Backpropagation
        if perceptual is not None:
            loss = image_loss + opt.perceptual_weight * perceptual_loss
        else:
            loss = image_loss

        loss.backward()
        optimizer.step()

        # Record the training loss
        trainloss.append(image_loss.item())
        if perceptual is not None:
            perceloss.append(perceptual_loss.item())

        # ----------------------------------------------------- #
        # Execute for a period                                  #
        # 1. Print the training message                         #
        # 2. Plot the gradient of each layer (Deprecated)       #
        # 3. Validate the model                                 #
        # 4. Saving the network                                 #
        # ----------------------------------------------------- #
        # 1. Print the training message
        if steps % opt.log_interval == 0:
            msg = "===> [Epoch {}] [{:4d}/{:4d}] ImgLoss: {:.6f}".format(
                epoch, iteration, len(train_loader), loss.item()
            )
        
            if not perceptual is None:
                msg = "\t".join([msg, "PerceptualLoss: {:.6f}".format(
                    perceptual_loss.item())]
                )

            print(msg)

        # 2. Print the gradient statistic message for each layer
        # graphs.draw_gradient()

        # 3. Save the model
        if steps % opt.save_interval == 0:
            checkpoint_path = os.path.join(opt.checkpoints, name, "{}.pth".format(steps))
            utils.saveCheckpoint(checkpoint_path, model, optimizer, scheduler, epoch, iteration)
        
        # 4. Validating the network
        if steps % opt.val_interval == 0:
            mse, psnr = validate(model, val_loader, criterion, epoch, iteration, normalize=opt.normalize)

            loss_iter = np.append(loss_iter, np.array([np.mean(trainloss)]), axis=0)
            mse_iter  = np.append(mse_iter, np.array([mse]), axis=0)
            psnr_iter = np.append(psnr_iter, np.array([psnr]), axis=0)
            lr_iter   = np.append(lr_iter, np.array([optimizer.param_groups[0]["lr"]]), axis=0)
            iters     = np.append(iters, np.array([steps / len(train_loader)]), axis=0)

            if perceptual is None:
                perc_iter = np.append(perc_iter, np.array([np.nan]), axis=0)

            if not perceptual is None:
                perc_iter = np.append(perc_iter, np.array([np.mean(perceloss)]), axis=0)

            # Clean up the list
            trainloss = []
            perceloss = []

            # Save the loss
            df = pd.DataFrame(data={
                'Iterations':      iters * len(train_loader),
                'TrainL2Loss':     loss_iter, 
                'TrainPerceptual': perc_iter,
                'ValidationLoss':  mse_iter, 
                'ValidationPSNR':  psnr_iter
            })

            # Loss (Training Curve) Message
            df = df.nlargest(5, 'ValidationPSNR').append(df)

            df.to_excel(os.path.join(opt.detail, name, "statistical.xlsx"))

            # Show images in grid with validation set
            # graphs.grid_show()
                
            # Plot TrainLoss, ValidationLoss
            fig, ax = training_curve(
                loss_iter, 
                perc_iter, 
                mse_iter, 
                psnr_iter, 
                ssim_iter, 
                iters, 
                lr_iter, 
                epoch, len(train_loader), 
                fig, ax
            )
            
            plt.tight_layout()
            plt.savefig(os.path.join(opt.detail, name, "loss.png"))

    return loss_iter, perc_iter, mse_iter, psnr_iter, ssim_iter, lr_iter, iters, fig, ax

def training_curve(train_loss, perc_iter, val_loss, psnr, ssim, x, lr, epoch, iters_per_epoch, fig: matplotlib.figure.Figure, axis: matplotlib.axes.Axes, linewidth=0.25):
    """
    Plot out learning rate, training loss, validation loss and PSNR.

    Parameters
    ----------
    train_loss, perc_iter, val_loss, psnr, ssim, lr, x: 1D-array like
        (...)

    iters_per_epoch : int
        To show the iterations in the epoch

    fig, axis : matplotlib.figure.Figure, matplotlib.axes.Axes
        Matplotlib plotting object.

    linewidth : float
        Default linewidth

    Return
    ------
    fig, axis : matplotlib.figure.Figure, matplotlib.axes.Axes
        The training curve
    """
    # Linear scale of loss curve
    ax = axis[0]
    line1, = ax.plot(x, val_loss, label="Validation Loss", color='red', linewidth=linewidth)
    line2, = ax.plot(x, train_loss, label="Train Loss", color='blue', linewidth=linewidth)
    ax.plot(x, np.repeat(np.amin(val_loss), len(x)), linestyle=':', linewidth=linewidth)
    ax.set_xlabel("Epoch(s) / Iteration: {}".format(iters_per_epoch))
    ax.set_ylabel("Image Loss")
    ax.set_title("Loss")

    if not np.isnan(perc_iter).all():
        ax = axis[4]
        line4, = ax.plot(x, perc_iter, label="Perceptual Loss", color='green', linewidth=linewidth)
        ax.set_ylabel("Perceptual Loss")

    if not np.isnan(perc_iter).all():
        ax.legend(handles=(line1, line2, line4, ))
    else:
        ax.legend(handles=(line1, line2, ))

    # Log scale of loss curve
    ax = axis[1]
    line1, = ax.plot(x, val_loss, label="Validation Loss", color='red', linewidth=linewidth)
    line2, = ax.plot(x, train_loss, label="Train Loss", color='blue', linewidth=linewidth)
    ax.plot(x, np.repeat(np.amin(val_loss), len(x)), linestyle=':', linewidth=linewidth)
    ax.set_xlabel("Epoch(s) / Iteration: {}".format(iters_per_epoch))
    ax.set_yscale('log')
    ax.set_title("Loss(Log scale)")

    if not np.isnan(perc_iter).all():
        ax = axis[5]
        line4, = ax.plot(x, perc_iter, label="Perceptual Loss", color='green', linewidth=linewidth)
        ax.set_ylabel("Perceptual Loss")

    if not np.isnan(perc_iter).all():
        ax.legend(handles=(line1, line2, line4, ))
    else:
        ax.legend(handles=(line1, line2, ))

    # Linear scale of PSNR, SSIM
    ax = axis[2]
    line1, = ax.plot(x, psnr, label="PSNR", color='blue', linewidth=linewidth)
    ax.plot(x, np.repeat(np.amax(psnr), len(x)), linestyle=':', linewidth=linewidth)
    ax.set_xlabel("Epochs(s) / Iteration: {}".format(iters_per_epoch))
    ax.set_ylabel("Average PSNR")
    ax.set_title("Validation Performance")

    ax.legend(handles=(line1, ))

    # Learning Rate Curve
    ax = axis[3]
    line1, = ax.plot(x, lr, label="Learning Rate", color='cyan', linewidth=linewidth)
    ax.set_xlabel("Epochs(s) / Iteration: {}".format(iters_per_epoch))
    ax.set_title("Learning Rate")
    ax.set_yscale('log')

    ax.legend(handles=(line1, ))
        
    return fig, axis

def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, epoch, iteration, normalize=False):
    """
    Validate the model

    Parameters
    ----------
    model : nn.Module
        The neural networks to train

    loader : torch.utils.data.DataLoader
        The training data
    
    epoch : int
        The training epoch
    
    criterion : nn.Module
        Loss function 
    
    normalize : bool
        If true, normalize the image before and after the NN.

    Return
    ------
    mse, psnr : np.float
        np.mean(mse) and np.mean(psnr)
    """
    psnrs, mses = [], []
    model.eval()

    with torch.no_grad():
        for index, (data, label) in enumerate(loader, 1):
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

            mses.append(mse)
            psnrs.append(psnr)

        print("[Vaild] Epoch: {}, MSE: {:.6f}, PSNR: {:.4f}".format(epoch, np.mean(mses), np.mean(psnrs)))

    return np.mean(mses), np.mean(psnrs)

if __name__ == "__main__":
    # Clean up OS screen
    os.system('clear')

    # Cmd Parser
    parser = cmdparser.parser
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

    # Check training dataset directory
    for path in opt.train:
        if not os.path.exists(path): 
            raise ValueError("{} doesn't exist".format(path))

    # Check validation dataset directory
    for path in opt.val:
        if not os.path.exists(path):
            raise ValueError("{} doesn't exist".format(path))

    # Make checkpoint storage directory
    name = "{}_{}".format(opt.tag, date.today().strftime("%Y%m%d"))
    os.makedirs(os.path.join(opt.checkpoints, name), exist_ok=True)

    # Copy the code of model to logging file
    if os.path.exists(os.path.join(opt.detail, name, 'model')):
        shutil.rmtree(os.path.join(opt.detail, name, 'model'))

    if os.path.exists(os.path.join(opt.checkpoints, name, 'model')):
        shutil.rmtree(os.path.join(opt.checkpoints, name, 'model'))

    shutil.copytree('./model', os.path.join(opt.detail, name, 'model'))
    shutil.copytree('./model', os.path.join(opt.checkpoints, name, 'model'))
    shutil.copyfile(__file__, os.path.join(opt.detail, name, os.path.basename(__file__)))

    # Show Detail
    print('==========> Training setting')
    utils.details(opt, os.path.join(opt.detail, name, 'args.txt'))

    # Execute main process
    main(opt)
