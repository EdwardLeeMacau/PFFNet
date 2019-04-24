# coding=utf-8
import argparse
import logging
import logging.config
import os
import pdb
# import urllib.request

import numpy as np

# pdb.set_trace()
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomCrop
# from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from skimage.measure import compare_psnr, compare_ssim
from matplotlib import pyplot as plt

from data import DatasetFromFolder
from model.rpnet import Net

from utils import save_checkpoint

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--tag", type=str, help="tag for this training")
parser.add_argument("--rb", type=int, default=18, help="number of residual blocks")
parser.add_argument("--train", default="./IndoorTrain", type=str, help="path to load train datasets")
parser.add_argument("--test", default="./IndoorTrain", type=str, help="path to load test datasets")
# parser.add_argument("--test", default="./dataset/IndoorTest", type=str, help="path to load test datasets")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=300, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=2000, help="step to test the model performance. Default=2000")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--resume", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--pretrained", type=str, help="path to pretrained model (default: none)")
# parser.add_argument("--report", default=False, type=bool, help="report to wechat")

# Set logger
logging.config.fileConfig("logging.ini")
statelogger = logging.getLogger(__name__)

def main():
    global opt, name, model, criterion
    # global opt, name, logger, model, criterion
    opt = parser.parse_args()
    print(opt)

    psnr_epochs = []
    ssim_epochs = []
    mse_epochs  = []
    epochs      = []
    train_epochs = []

    # Tag_ResidualBlocks_BatchSize
    name = "%s_%d_%d" % (opt.tag, opt.rb, opt.batchSize)

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

    indoor_test_dataset = DatasetFromFolder(opt.test, transform=Compose([
        ToTensor()
    ]))

    training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True, shuffle=True)
    indoor_test_loader = DataLoader(dataset=indoor_test_dataset, num_workers=opt.threads, batch_size=1, pin_memory=True, shuffle=True)

    print("==========> Building model")
    model = Net(opt.rb)
    criterion = nn.MSELoss(size_average=True)

    # print(model)

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

    print("==========> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        trainloss = train(training_data_loader, indoor_test_loader, optimizer, epoch)
        mse, psnr = test(indoor_test_loader, epoch)
        
        train_epochs.append(trainloss)
        mse_epochs.append(mse)
        psnr_epochs.append(psnr)
        # ssim_epochs.append(ssim)
        epochs.append(epoch)
        
        if psnr > max(psnr_epochs):
            save_checkpoint(model, epoch, name)
            # test(indoor_test_loader, epoch)

        # Plot TrainLoss
        plt.clf()
        plt.plot(epochs, train_epochs, label="TrainLoss")
        plt.xlabel("Epoch(s)")
        plt.legend(loc=0)
        plt.title("TrainLoss vs Epochs")
        plt.savefig("TrainLoss.png")

        # Plot MSE
        plt.clf()
        plt.plot(epochs, mse_epochs, label="MSE")
        plt.xlabel("Epoch(s)")
        plt.legend(loc=0)
        plt.title("MSE vs Epochs")
        plt.savefig("MSE.png")
        
        # Plot PSNR and SSIM
        plt.clf()
        plt.plot(epochs, psnr_epochs, label="PSNR vs Epochs")
        plt.xlabel("Epoch(s)")
        plt.legend(loc=0)
        plt.title("PSNR vs Epochs")
        plt.savefig("PSNR.png")

        # fig, axis1 = plt.subplots()
        # axis1.set_xlabel('Epoch(s)')
        # axis1.set_ylabel('Average PSNR')
        # axis2 = axis1.twinx()
        # axis2.set_ylabel('Average SSIM')

        # axis1.plot(epochs, psnr_epochs, label="PSNR vs Epochs")
        # axis2.plot(epochs, ssim_epochs, label="SSIM vs Epochs")
        # plt.legend(loc=0)
        # plt.title("PSNR-SSIM vs Epochs")
        # fig.tight_layout()
        # fig.savefig("PSNR-SSIM.png")

def train(training_data_loader, indoor_test_loader, optimizer, epoch):
    statelogger.info("epoch: {}, lr: {}".format(epoch, optimizer.param_groups[0]["lr"]))
    # print("Memory Usage: {}".format(torch.cuda.memory_allocated(0)))
    
    trainLoss = []

    for iteration, batch in enumerate(training_data_loader, 1):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        steps = len(training_data_loader) * (epoch - 1) + iteration

        data, label = batch[0].cuda(), batch[1].cuda()

        # if opt.cuda:
        data = data.cuda()
        label = label.cuda()
        
        # else:
        #     data = data.cpu()
        #     label = label.cpu()

        output = model(data)

        # loss = criterion(output, label) / (data.size()[0]*2)
        loss = criterion(output, label)
        loss.backward()

        trainLoss.append(loss.item())

        # torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
        optimizer.step()

        if iteration % 10 == 0:
            #print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
            statelogger.info("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader), loss.item()))
            # logger.add_scalar('loss', loss.data[0], steps)

        """
        if iteration % opt.step == 0:
            data_temp = make_grid(data.data)
            label_temp = make_grid(label.data)
            output_temp = make_grid(output.data)

            # logger.add_image('data_temp', data_temp, steps)
            # logger.add_image('label_temp', label_temp, steps)
            # logger.add_image('output_temp', output_temp, steps)
        """

    trainLoss = np.asarray(trainLoss)
    return np.mean(trainLoss)

def test(test_data_loader, epoch):
    psnrs = []
    ssims = []
    mses = []
    model.eval()

    for iteration, batch in enumerate(test_data_loader, 1):
        data, label = Variable(batch[0], volatile=True), Variable(batch[1])

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()

        with torch.no_grad():
            output = model(data)

        output = torch.clamp(output, 0., 1.)
        
        mse = nn.MSELoss()(output, label)
        mses.append(mse.item())
        psnr = 10 * np.log10(1.0 / mse.item())
        psnrs.append(psnr)
        # ssim = 0
        # for i in range(output.shape[0]):
        #     ssim += compare_ssim(output[i], label[i], multichannel=True)
        # ssim /= output.shape[0]
        # ssim = compare_ssim(output, label)
        # ssims.append(ssim)

        # TODO: Use library of PSNR and SSIM instead.
    
    psnr_mean = np.mean(psnrs)
    mse_mean  = np.mean(mses)
    # ssim_mean = np.mean(ssims)

    # print("Vaild  epoch %d psnr: %f" % (epoch, psnr_mean))
    statelogger.info("[Vaild] epoch: {}, psnr: {}".format(epoch, psnr_mean))
    # statelogger.info("[Vaild] epoch: {}, psnr: {}".format(epoch, ssim_mean))
    # logger.add_scalar('psnr', psnr_mean, epoch)
    # logger.add_scalar('mse', mse_mean, epoch)

    data = make_grid(data.data)
    label = make_grid(label.data)
    output = make_grid(output.data)

    # logger.add_image('data', data, epoch)
    # logger.add_image('label', label, epoch)
    # logger.add_image('output', output, epoch)

    return mse_mean, psnr_mean

if __name__ == "__main__":
    os.system('clear')
    main()
