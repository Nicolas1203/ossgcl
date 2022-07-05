"""Custom CIFAR100 implementation to load one class at a time only.
"""
import torch

from src.utils.utils import filter_labels
from torchvision.datasets.cifar import CIFAR10
from PIL import Image


class SplitCIFAR10(CIFAR10):
    def __init__(self, root, train, transform, download=False, number_list=[0]):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.number_list = number_list
        self.targets = torch.Tensor(self.targets)
        self.indexes = torch.nonzero(filter_labels(self.targets, self.number_list)).flatten()

    def __getitem__(self, index):
        img, target = self.data[self.indexes[index]], self.targets[self.indexes[index]]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indexes)
