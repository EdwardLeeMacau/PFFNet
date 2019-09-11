"""
  FileName     [ data.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Provide definition of dataset. ]
"""

import argparse
import os
import time
from os.path import basename

import torch.utils.data as data
import torchvision.transforms
from PIL import Image
from torch.utils.data import DataLoader

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        data_dir  = os.path.join(image_dir, "Hazy")
        label_dir = os.path.join(image_dir, "Gt")
        self.data_filenames  = sorted([os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_image_file(x)])
        self.label_filenames = sorted([os.path.join(label_dir, x) for x in os.listdir(label_dir) if is_image_file(x)])

        self.transform = transform

    def __getitem__(self, index):
        data  = Image.open(self.data_filenames[index])
        label = Image.open(self.label_filenames[index])

        if self.transform:
            data  = self.transform(data)
            label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)
