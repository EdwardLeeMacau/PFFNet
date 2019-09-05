import math
import os
from os import listdir
from os.path import join

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

def selectDevice():
    """ Return the device available in computer """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def weights_init_kaiming(m):
    """ 
    Kaiming weights initial methods

    Parameters
    ----------
    m : torch.nn.Module
        the models 
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)

def output_psnr_mse(origin, gen):
    """ 
    Measure the PSNR and MSE of 2 pictures 
    
    Parameters
    ----------
    origin, gen : np.ndarray
        The ground truth and generated images
    """
    squared_error = np.square(origin - gen)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)

    return psnr


def is_image_file(filename):
    """ Return true if the file is an image. """
    filename_lower = filename.lower()

    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

# Read the images in 1 folder.
def load_all_image(path):
    return [ join(path, x) for x in listdir(path) if is_image_file(x) ]

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


def get_mean_and_std(dataset: torch.utils.data.Dataset):
    """
    Return the mean and std value of dataset

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        dataset instance

    Return
    ------
    mean, std
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def saveCheckpoint(checkpoint_path, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.MultiStepLR, epoch, iteration):
    """
      Params:
      - checkpoint_path: the directory of the model parameter
      - model: the neural network to save
      - optimizer
      - scheduler

      Return: model, optimizer, resume_epoch, resume_iteration, scheduler
    """
    # state_dict = model.module.state_dict()
    # for key in state_dict.keys():
    #     state_dict[key] = state_dict[key].cpu()

    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
        'scheduler': scheduler.state_dict()
    }

    torch.save(state, checkpoint_path)

    return

def loadCheckpoint(checkpoint_path: str, model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler.MultiStepLR):
    """
      Params:
      - checkpoint_path: the directory of the model parameter
      - model: the neural network to save
      - optimizer
      - scheduler

      Return: model, optimizer, resume_epoch, resume_iteration, scheduler
    """
    state = torch.load(checkpoint_path)

    resume_epoch = state['epoch']
    resume_iteration = state['iteration']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    return model, optimizer, resume_epoch, resume_iteration, scheduler

def saveModel(checkpoint_path: str, model: nn.Module):
    """
    Save the model's parameters

    Parameters
    ----------
    checkpoint_path : str
        the directory of the model parameter
    
    model : torch.nn.Module
        the neural network to save
    """
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)

    return

def loadModel(checkpoint_path: str, model: nn.Module, dataparallel=False):
    """
    Load the model parameters
    
    Parameters
    ----------
    checkpoint_path : str
        the directory of the model parameter

    model : torch.nn.Module
        the neural network to load
    
    dataparallel : 
        If true, the key of the state_dict will have a 'module' prefix, remove it. 
    """
    state = torch.load(checkpoint_path)
    
    if dataparallel:
        for key in list(state['state_dict'].keys()):
            newkey = ".".join(key.split('.')[1:])
            state['state_dict'].update({newkey: state['state_dict'].pop(key)})
    
    model.load_state_dict(state['state_dict'])

    return model

def checkpointToModel(checkpoint_path: str, model_path: str):
    """
    Return the model parameters in the checkpoint file

    Parameters
    ----------
    checkpoint : str
        Includes model, optimizer, schduler, save epochs and iterations.
    
    model : str
        Includes model parameters only
    """
    state = torch.load(checkpoint_path)
    newState = {'state_dict': state['state_dict']}

    torch.save(newState, model_path)

    return

def details(opt, path=None):
    """
    Show and marked down the training settings

    Parameters
    ----------
    opt : namespace
        The namespace of the train setting
    
    path : str, optional
        the path to output textfile
    """
    makedirs = []

    if path is not None:        
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        with open(path, "w") as textfile:
            for item, values in vars(opt).items():
                msg = "{:16} {}".format(item, values)
                textfile.write(msg + '\n')
    
    for item, values in vars(opt).items():
        print("{:16} {}".format(item, values))
            
    return
