"""
  FileName    [ train_improve.py }
  PackageName [ PFFNet ]
  Synopsis    [ Train the model with improved structure ]
"""

import argparse
import os
import shutil
from datetime import date

import utils
import cmdparser
from model.rpnet import Net
from model.rpnet import ImproveNet
from train import getDataset, getOptimizer, getTrainSpec, train 

device = utils.selevtDevice()
cudnn.benchmark = True

def main(opt):
    """
    Main process of train_improve.py

    Parameters
    ----------
    opt : namespace
        The option (hyperparameters) of these model
    """

    model, train_loader, val_loader, optimizer, schedular, criterion, _ = getTrainSpec(opt)

    for i in range(1, opt.epochs + 1):    
        model, optimizer, schedular, criterion, train_loader, val_loader = train(model, optimizer, schedular, criterion, train_loader, val_loader)

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

    # Copy the code of model to logging file
    if os.path.exists(os.path.join(opt.detail, name, 'model')):
        shutil.rmtree(os.path.join(opt.detail, name, 'model'))

    if os.path.exists(os.path.join(opt.checkpoints, name, 'model')):
        shuitl.rmtree(os.path.join(opt.checkpoints, name, 'model'))

    shutil.copytree('./model', os.path.join(opt.detail, name, 'model'))
    shutil.copytree('./model', os.path.join(opt.checkpoints, name, 'model'))
    shutil.copyfile(__file__, os.path.join(opt.detail, name, os.path.basename(__file__)))

    print('==========> Training setting')
    utils.details(opt, os.path.join(opt.detail, name, 'args.txt'))

    main()
