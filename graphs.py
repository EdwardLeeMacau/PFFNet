"""
  FileName     [ graphs.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Provide statistics function for Deep Learning. ]
"""
import trochvision
from matplotlib import pyplot as plt

def draw_gradient():
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
      Params:
      - model:
      - loader:
      - folder:
      - nrow:
      - normalize:

      Return: None
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

    num_curves = len(labels)
    for i in range(0, num_curves):
        plt.clf()
        
        if 'figsize' in kwargs: 
            plt.figure(figsize=figsize)
        
        plt.plot(x, y[i], label=label[i])

        plt.legend(loc=0)
        plt.title(titles[i])
        plt.savefig(filenames[i])

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

    return