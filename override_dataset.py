import torch
from torch.utils import data
import torchvision


class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CustomCIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.map_to_val = dict()

    def __getitem__(self, index):
        image, target = super(CustomCIFAR10, self).__getitem__(index)
        return image, self.map_to_val.get(index, target)
    
    def change_label(self, index, label):
        self.map_to_val[index] = label


class CustomSubset(data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.map_to_val = dict()

    def __getitem__(self, index):
        image, target = self.dataset[self.indices[index]]
        return image, self.map_to_val.get(index, target)

    def change_label(self, index, label):
        self.map_to_val[index] = label


class CustomConcatDataset(data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.map_to_val = dict()

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, self.map_to_val.get(index, target)
       
    def change_label(self, index, label):
        self.map_to_val[index] = label
