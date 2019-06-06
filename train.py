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
parser.add_argument("--rb", default=18, type=int, help="number of residual blocks")
parser.add_argument("--batchsize", default=16, type=int, help="training batch size")
parser.add_argument("--epochs", default=15, type=int, help="number of epochs to train for")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning Rate. Default=1e-4")
parser.add_argument("--activation", default="LeakyReLU", type=str, help="the activation function use at training")
parser.add_argument("--normalize", default=False, action="store_true", help="normalized the dataset images")
parser.add_argument("--milestones", default=[10], type=int, nargs='*', help="Which epoch to decay the learning rate")
parser.add_argument("--gamma", default=0.1, type=float, help="The ratio of decaying learning rate everytime")
parser.add_argument("--starts", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--pretrained", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="The weight penalty in the training")
parser.add_argument("--optimizer", default="Adam", type=str, help="Choose the optimizer")
parser.add_argument("--loss_function", default=['L2'], type=str, nargs='*', help="loss function used in training, [l1, l2, preceptual] is allow")
# Message logging, model saving setting
parser.add_argument("--tag", default="Indoor_512", type=str, help="tag for this training")
parser.add_argument("--checkpoints", default="/media/disk1/EdwardLee/checkpoints", type=str, help="path to save the checkpoints")
parser.add_argument("--val_interval", default=1000, type=int,  help="step to test the model performance")
parser.add_argument("--log_interval", default=10, type=int, help="interval per iterations to log the message")
parser.add_argument("--grad_interval", default=0, type=int, help="interval per iterations to draw the gradient")
parser.add_argument("--save_interval", default=1000, type=int, help="interval per iterations to save the model")
parser.add_argument("--detail", default="./log", type=str, help="the root directory to save the training details")
# Device setting
parser.add_argument("--cuda", default=True, type=bool, help="Use cuda?")
parser.add_argument("--gpus", default=1, type=int, help="nums of gpu to use")
parser.add_argument("--threads", default=8, type=int, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--fixrandomseed", default=False, help="train with fix random seed")
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

# Select Device
device = utils.selectDevice()
cudnn.benchmark = True

mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
std  = torch.Tensor([0.229, 0.224, 0.225]).to(device)

# ----------------------------------------------------------------------------
# Normalization (mean shift)
# ----------------------------------------------------------------------------
# Normalization methods
#   input = (input - mean[:, None, None]) / std[:, None, None]
#
# How to inverse:
#   input = (input * std[:, None, None]) + mean[:, None, None]
# 
# Source code:
#   tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
#
# Pretrain network normalize parameterss
#   mean = [0.485, 0.456, 0.406]
#   std  = [0.229, 0.224, 0.225]
# ----------------------------------------------------------------------------

def main(opt):
    name = "{}_{}_{}_{}".format(opt.tag, date.today().strftime("%Y%m%d"), opt.rb, opt.batchsize)
    
    if opt.fixrandomseed:
        seed = 1334
        torch.manual_seed(seed)
        
        if opt.cuda: 
            torch.cuda.manual_seed(seed)

    print("==========> Loading datasets")
    transform = [ToTensor()]
    if opt.normalize: transform.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    img_transform = Compose(transform)


    train_dataset = DatasetFromFolder(opt.train, transform=img_transform)
    val_dataset   = DatasetFromFolder(opt.val, transform=img_transform)
    train_loader  = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchsize, pin_memory=True, shuffle=True)
    val_loader    = DataLoader(dataset=val_dataset, num_workers=opt.threads, batch_size=opt.batchsize, pin_memory=True, shuffle=True)
    
    print("==========> Building model")
    # ----------------------------------------------------------------------------
    # Notes: 20190515
    #   The original model doesn't set any activation function in the output layer.
    # -----------------------------------------------------------------------------
    model = Net(opt.rb)
    
    # --------------------------------------
    # Loss
    # -> opt.loss_function
    # -> L1 Norm / L2 Norm / Perceptual loss 
    # --------------------------------------
    criterion = nn.MSELoss(size_average=True)
    
    # vgg16 = ...

    # --------------------------------------
    # Optimizer and learning rate scheduler
    # --------------------------------------
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

    # -------------------------
    # Load the network
    # -------------------------
    if opt.resume and opt.pretrained:
        raise argparse.ArgumentError

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            model, optimizer, opt.starts, opt.iterations, scheduler = utils.loadCheckpoint(opt.resume, model, optimizer, scheduler)
        else:
            raise Exception("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading pretrained model '{}'".format(opt.pretrained))
            model = utils.loadModel(opt.pretrained, model)
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

    # Extablish container
    loss_iter  = np.empty(0, dtype=float)
    psnr_iter  = np.empty(0, dtype=float)
    ssim_iter  = np.empty(0, dtype=float)
    mse_iter   = np.empty(0, dtype=float)
    iterations = np.empty(0, dtype=float)

    print("==========> Training setting")
    details(opt, "./{}/{}/{}".format(opt.detail, name, "args.txt"))

    print("==========> Training")
    for epoch in range(opt.starts, opt.epochs + 1):
        scheduler.step()

        loss_iter, mse_iter, psnr_iter, ssim_iter, iterations = train_val(
            model, optimizer, criterion, train_loader, val_loader, scheduler, 
            epoch, loss_iter, mse_iter, psnr_iter, ssim_iter, iterations, opt, name
        )

    return

def train_val(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                   scheduler: optim.lr_scheduler.MultiStepLR, epoch, loss_iter, mse_iter, psnr_iter, ssim_iter, iters, opt, name):
    print("===> lr: ", optimizer.param_groups[0]["lr"])
    
    trainLoss = []

    for iteration, (data, label) in enumerate(train_loader, 1):
        # ------------------
        # Train the network
        # ------------------
        model.train()
        optimizer.zero_grad()

        steps = len(train_loader) * (epoch - 1) + iteration

        data, label = data.to(device), label.to(device)

        output = model(data)
        loss = criterion(output, label)
        loss.backward()

        trainLoss.append(loss.item())
        optimizer.step()

        # ------------------------------------------------------------------------------
        # 1. Log the training message
        # 2. Plot the gradient of each layer
        # 3. Validate the model
        # 4. Saving the network
        # ------------------------------------------------------------------------------
        # 1. Log the training message
        if steps % opt.log_interval == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(train_loader), loss.item()))
        
        # 2. Plot the gradient of each layer
        if opt.grad_interval:
            # (Deprecated)
            if steps % opt.grad_interval == 0:
                layer_names, mean, abs_mean, std = [], [], [], []

                for layer_name, param in model.named_parameters():
                    layer_names.append('.'.join(layer_name.split('.')[1:]))
                    
                    values = param.grad.detach().view(-1).cpu().numpy()
                    mean.append(np.mean(values))
                    abs_mean.append(np.mean(np.absolute(values)))
                    std.append(np.std(values))
                
                plt.clf()
                plt.figure(figsize=(19.2, 10.8))
                plt.subplot(3, 1, 1)
                plt.bar(np.arange(len(std)), np.asarray(std), 0.5)
                plt.title("STD vs layer")
                plt.subplot(3, 1, 2)
                plt.bar(np.arange(len(mean)), np.asarray(mean), 0.5)
                plt.title("Mean vs layer")
                plt.subplot(3, 1, 3)
                plt.bar(np.arange(len(abs_mean)), np.asarray(abs_mean), 0.5)
                plt.title("Mean(Abs()) vs layer")
                # plt.savefig("./{}/{}/grad_std_{}.png".format(opt.detail, name, str(steps).zfill(len(str(opt.nEpochs * len(train_loader))))))
                # plt.table(rowLabels=["Mean", "STD"], 
                #         colLabels=layer_names,
                #         cellText=np.asarray([mean, std], dtype=np.float32))
                plt.savefig("./{}/{}/grad_{}.png".format(opt.detail, name, str(steps).zfill(len(str(opt.nEpochs * len(train_loader))))))
        
        # 3. Validate the model
        if steps % opt.save_interval == 0:
            # In epoch testing and saving
            utils.saveCheckpoint(opt.checkpoints, model, optimizer, scheduler, epoch, iteration)
        
        # 4. Saving the network
        if steps % opt.val_interval == 0:
            # mses, psnrs, ssims = test(val_loader, epoch, criterion)
            mse, psnr = validate(model, val_loader, criterion, epoch, iteration, normalize=opt.normalize)
            loss_iter = np.append(loss_iter, np.array([np.mean(trainLoss)]), axis=0)
            mse_iter  = np.append(mse_iter, np.array([mse]), axis=0)
            psnr_iter = np.append(psnr_iter, np.array([psnr]), axis=0)
            iters     = np.append(iters, np.array([steps / len(train_loader)]), axis=0)

            trainLoss = []  # Clean the list 
            
            with open(os.path.join(opt.detail, name, "statistics.txt"), "w") as textfile:
                datas = [str(data.tolist()) for data in (loss_iter, mse_iter, psnr_iter, ssim_iter, iters)]
                textfile.write("\n".join(datas))
                
            # Plot TrainLoss, valloss
            draw_graphs(loss_iter, mse_iter, psnr_iter, ssim_iter, iters, len(train_loader), os.path.join(opt.detail, name))

    return loss_iter, mse_iter, psnr_iter, ssim_iter, iters

def details(opt, path):
    """
      Show and marked down the training settings

      Params:
      - opt: The namespace of the train setting (Usually argparser)
      - path: the path output textfile

      Return: None
    """
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
            textfile.write(msg + '\n')
    
    return

def draw_graphs(train_loss, val_loss, psnr, ssim, x, iters_per_epoch, savepath,
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
    plt.savefig(os.path.join(savepath, loss_filename))

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
    plt.savefig(os.path.join(savepath, loss_log_filename))

    # Linear scale of PSNR, SSIM
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    
    plt.plot(x, psnr, label="PSNR", color='b')
    plt.plot(x, np.repeat(np.amax(psnr), len(x)), ':')
    plt.xlabel("Epochs(s) / Iteration: {}".format(iters_per_epoch))
    plt.title("PSNR vs Epochs")
        
    plt.legend(loc=0)
    plt.savefig(os.path.join(savepath, psnr_filename))

    return

def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, epoch, iteration, normalize=False):
    """
      Params:
      - model
      - loader
      - epoch
      - criterion
      - normalize

      Return:
      - np.mean(mse)
      - np.mean(psnr)
    """
    psnrs, mses = [], []
    model.eval()

    with torch.no_grad():
        for (data, label) in tqdm(loader):
            data, label = data.to(device), label.to(device)

            output = model(data)
            mse = criterion(output, label).item()
            mses.append(mse)
            
            # (batchsize, width, height, channel)
            output = output.permute(0, 2, 3, 1).cpu().numpy()
            label  = label.permute(0, 2, 3, 1).cpu().numpy()

            psnr = 10 * np.log10(1.0 / mse)
            psnrs.append(psnr)


        # Revert to domain [0, 1]
        if normalize:
            data   = data * std[:, None, None] + mean[:, None, None]
            label  = label * std[:, None, None] + mean[:, None, None]
            output = output * std[:, None, None] + mean[:, None, None]

        # Random sample 8 images from dataloader, draw the output
        data_temp   = make_grid(data.data, nrow=8)
        label_temp  = make_grid(label.data, nrow=8)
        output_temp = make_grid(output.data, nrow=8)

        torchvision.utils.save_image(data_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_data.png".format(epoch, iteration))
        torchvision.utils.save_image(label_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_label.png".format(epoch, iteration))
        torchvision.utils.save_image(output_temp, "/media/disk1/EdwardLee/images/Image_{}_{}_output.png".format(epoch, iteration))

        print("[Vaild] epoch: {}, mse:  {}".format(epoch, np.mean(mse)))
        print("[Vaild] epoch: {}, psnr: {}".format(epoch, np.mean(psnr)))

    return np.mean(mses), np.mean(psnrs)

if __name__ == "__main__":
    os.system('clear')

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    main(opt)
