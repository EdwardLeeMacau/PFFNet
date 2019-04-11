import os
import argparse
import time
from PIL import Image
from os.path import basename
import torch.utils.data as data
import torchvision.transforms
from torch.utils.data import DataLoader

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--train", default="./IndoorTrain", type=str, help="path to load train datasets")
parser.add_argument("--test", default="./IndoorTrain ", type=str, help="path to load test datasets")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--threads", default=4, help="Number of threads for data loader to use")
args = parser.parse_args()

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        data_dir  = os.path.join(image_dir, "data")
        label_dir = os.path.join(image_dir, "label")
        self.data_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_image_file(x)]
        self.label_filenames = [os.path.join(label_dir, x) for x in os.listdir(label_dir) if is_image_file(x)]

        self.transform = transform

    def __getitem__(self, index):
        data = Image.open(self.data_filenames[index])
        label = Image.open(self.label_filenames[index])

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)

def unittest():
    training_dataset = DatasetFromFolder(args.train, transform=torchvision.transforms.ToTensor())
    testing_dataset  = DatasetFromFolder(args.test, transform=torchvision.transforms.ToTensor())

    training_loader = DataLoader(dataset=training_dataset, num_workers=args.threads, batch_size=args.batchSize, pin_memory=True, shuffle=True)
    testing_loader  = DataLoader(dataset=testing_dataset, num_workers=args.threads, batch_size=1, pin_memory=True, shuffle=False)

    # Test loading speed
    train_iter = iter(training_loader)
    start = time.time()
    for i in range(0, args.batchSize):
        data, label = next(train_iter)

    print("Using time: {}".format(time.time() - start))

if __name__ == "__main__":
    unittest()