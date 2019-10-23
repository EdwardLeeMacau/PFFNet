"""
  FileName     [ cmaparse.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Common commands for the package ]
"""

import argparse

parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")

# Basic Training settings
parser.add_argument("--rb", default=18, type=int, help="number of residual blocks")
parser.add_argument("--batchsize", default=16, type=int, help="training batch size")
parser.add_argument("--epochs", default=100, type=int, help="number of epochs to train for")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning Rate. Default=1e-4")
parser.add_argument("--perceptual", type=str, help="Perceptual loss model selection")
parser.add_argument("--perceptual_weight", type=float, help="Weight of perceptual loss")
# parser.add_argument("--activation", default="LeakyReLU", type=str, help="the activation function use at training")
parser.add_argument("--normalize", action="store_true", help="normalized the dataset images")
parser.add_argument("--milestones", default=[20, 40], type=int, nargs='*', help="Which epoch to decay the learning rate")
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
parser.add_argument("--save_item", default="model", type=str, choices=("model", "checkpoint"), help="Option: {model, checkpoint}")
parser.add_argument("--detail", default="./log", type=str, help="the directory to save the training settings")

# Device setting
parser.add_argument("--cuda", default=True, action='store_true', help="Use cuda?")
parser.add_argument("--gpus", default=1, type=int, help="nums of gpu to use")
parser.add_argument("--threads", default=8, type=int, help="Number of threads for data loader to use.")
parser.add_argument("--fixrandomseed", default=False, help="train with fix random seed")

# Pretrain model setting
parser.add_argument("--resume", type=str, help="Path to checkpoint.")

# Dataloader setting
parser.add_argument("--train", default=["./dataset/NTIRE2018"], type=str, nargs='+', help="path of training dataset")
parser.add_argument("--val", default=["./dataset/NTIRE2018_VAL"], type=str, nargs='+', help="path of validation dataset")

if __name__ == "__main__":
    import utils

    opt = parser.parse_args()
    utils.details(opt, None)
