"""
  FileName     [ graphs.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Provide statistics function for Deep Learning. ]
"""
import torch
import torch.nn as nn 
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def draw_gradient():
    """ Plot the gradient of the training process """
    # if opt.grad_interval:
        # (Deprecated)
        # if steps % opt.grad_interval == 0:
        #     layer_names, mean, abs_mean, std = [], [], [], []

        #     for layer_name, param in model.named_parameters():
        #         layer_names.append('.'.join(layer_name.split('.')[1:]))
                
        #         values = param.grad.detach().view(-1).cpu().numpy()
        #         mean.append(np.mean(values))
        #         abs_mean.append(np.mean(np.absolute(values)))
        #         std.append(np.std(values))
            
        #     plt.clf()
        #     plt.figure(figsize=(19.2, 10.8))
        #     plt.subplot(3, 1, 1)
        #     plt.bar(np.arange(len(std)), np.asarray(std), 0.5)
        #     plt.title("STD vs layer")
        #     plt.subplot(3, 1, 2)
        #     plt.bar(np.arange(len(mean)), np.asarray(mean), 0.5)
        #     plt.title("Mean vs layer")
        #     plt.subplot(3, 1, 3)
        #     plt.bar(np.arange(len(abs_mean)), np.asarray(abs_mean), 0.5)
        #     plt.title("Mean(Abs()) vs layer")
        #     plt.savefig("./{}/{}/grad_{}.png".format(opt.detail, name, str(steps).zfill(len(str(opt.nEpochs * len(train_loader))))))

    return    

def grid_show(model: nn.Module, loader: DataLoader, folder, nrow=8, normalize=False):
    """
    Grid show function

    Parameters
    ----------
    model : nn.Module
        (...)

    loader : DataLoader
        (...)

    folder : str
        (...)

    nrow : int
        Default 8

    normalize : bool
        Default False
    """
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

def draw_graphs(x, y, labels, titles, filenames, **kwargs):
    """
    Parameters
    ----------
    x :
    
    y :
    
    labels :
    """
    if len(y) != len(labels):
        raise ValueError("The lengths of the labels should equal to the length of y.")

    return
