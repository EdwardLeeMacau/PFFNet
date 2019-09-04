import time

from tqdm import tqdm

import torchvision
from data import DatasetFromFolder
from torch.utils.data import DataLoader


def dataloader_unittest(train, test, threads=8, batchsize=16):
    """ Test loading I/O, dataloader speed. """
    training_dataset = DatasetFromFolder(train, transform=torchvision.transforms.ToTensor())
    testing_dataset  = DatasetFromFolder(test, transform=torchvision.transforms.ToTensor())

    training_loader = DataLoader(dataset=training_dataset, num_workers=threads, batch_size=batchsize, pin_memory=True, shuffle=True)
    testing_loader  = DataLoader(dataset=testing_dataset, num_workers=threads, batch_size=1, pin_memory=True, shuffle=False)

    # Test loading speed
    start = time.time()
    for data, label in tqdm(training_loader):
        pass

    print("Using time: {}".format(time.time() - start))

def main():
    # dataloader_unittest()
    pass

if __name__ == "__main__":
    main()
