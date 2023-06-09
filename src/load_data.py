import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split
import os
from src import _DATA_PATH


def get_dataloaders(batch_size, num_workers=8):
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.GaussianBlur(kernel_size=5),
            transforms.Resize((128, 128), antialias=None)
        ]
    )
    data_transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((128, 128), antialias=None)
        ]
    )
    # load train data from "/u/data/s194333/DLCV/project1_02514/data/train/hotdog"
    trainset = datasets.ImageFolder(
        root=os.path.join(_DATA_PATH, "train"), transform=data_transform
    )
    testset = datasets.ImageFolder(
        root=os.path.join(_DATA_PATH, "test"), transform=data_transform
    )
    testset2 = datasets.ImageFolder(
        root=os.path.join(_DATA_PATH, "test"), transform=data_transform_test
    )
    generator1 = torch.Generator().manual_seed(42)
    valset_new, testset_new = random_split(testset, [0.5, 0.5], generator=generator1)
    valset_new2, testset_new2 = random_split(testset, [0.5, 0.5], generator=generator1)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=8, shuffle=True
    )
    valloader = DataLoader(valset_new, batch_size=batch_size, num_workers=8)
    testloader = DataLoader(testset_new2, batch_size=batch_size, num_workers=8)

    return trainloader, valloader, testloader
