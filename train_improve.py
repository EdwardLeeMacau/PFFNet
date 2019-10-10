"""
  FileName    [ train_improve.py }
  PackageName [ PFFNet ]
  Synopsis    [ Train the model with improved structure ]
"""

import argparse
import os

from model.rpnet import Net
import train
import utils
import cmdparser

device = utils.selevtDevice()
cudnn.benchmark = True

# mean = utils.mean.to(device)
# std  = utils.std.to(device)

def main():
    """
    Main process of train_improve.py

    Parameters
    ----------
    opt : namespace
        The option (hyperparameters) of these model
    """

    return

if __name__ == '__main__':
    # Clean up OS screen
    os.system('clear')

    # Training setting
    parser = cmdparser.parser
    opt = parser.parse_args()

    # Check arugments
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

    main()
