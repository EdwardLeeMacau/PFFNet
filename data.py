"""
  FileName     [ data.py ]
  PackageName  [ PFFNet ]
  Synopsis     [ Definition of dataset. ]
"""

import os
from collections.abc import Container

import torch.utils.data as data
from PIL import Image, ImageFile

# Set True to load truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_image_file(filename):
    """ 
    Return true if the file is an image 

    Parameters
    ----------
    filename : str
        the name of the image file

    Return
    ------
    bool : bool
        True if **file** is an image.
    """
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Parameters
        ----------
        image_dir : { str, list-like }
            Dataset directory

        transform : torchvision.transforms
            Transform function of torchvision.
        """
        super(DatasetFromFolder, self).__init__()

        self.data_filenames  = []
        self.label_filenames = []

        # Modify Parameters
        if isinstance(image_dir, str):
            image_dir = (image_dir, )

        if not isinstance(image_dir, Container):
            raise ValueError("Image Directory should be type 'str' or type 'Container'")

        # Get File names
        for directory in image_dir:
            data_dir  = os.path.join(directory, "Hazy")
            label_dir = os.path.join(directory, "GT")
            self.data_filenames.extend( sorted([ os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_image_file(x) ]) )
            self.label_filenames.extend( sorted([ os.path.join(label_dir, x) for x in os.listdir(label_dir) if is_image_file(x) ]) )

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

def main():
    import cmdparser
    import torch
    import torchvision
    """ Check to load whole dataset. """
    opt = cmdparser.parser.parse_args()
    
    for dirs in (opt.train, opt.val):
        for directory in dirs:
            loader = torch.utils.data.DataLoader(
                DatasetFromFolder(directory, transform=torchvision.transforms.ToTensor()),
                batch_size=12, num_workers=8
            )

            for index, (hazy, gt) in enumerate(loader, 1):
                if (index % 100 == 0): print(directory, "{:4d} / {:4d}".format(index, len(loader)))

    return 

if __name__ == "__main__":
    main()
