import os

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import condensa.data

def cifar_train_val_loader(dataset,
                           train_batch_size,
                           val_batch_size,
                           root='./data',
                           random_seed=42,
                           shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    trainset = dataset(root=root,
                       train=True,
                       download=True,
                       transform=transform_train)
    valset = dataset(root=root, train=True, download=True, transform=None)
    num_train = len(trainset)
    indices = list(range(num_train))
    split = 5000

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    trainsampler = SubsetRandomSampler(train_idx)
    valsampler = SubsetRandomSampler(val_idx)

    meanstd = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trainloader = condensa.data.GPUDataLoader(trainset,
                                              batch_size=train_batch_size,
                                              shuffle=False,
                                              num_workers=8,
                                              sampler=trainsampler,
                                              meanstd=meanstd)
    valloader =   condensa.data.GPUDataLoader(valset,
                                              batch_size=val_batch_size,
                                              shuffle=False,
                                              num_workers=8,
                                              sampler=valsampler,
                                              meanstd=meanstd)

    return (trainloader, valloader)

def cifar_test_loader(dataset, batch_size, root='./data'):

    
    testset = dataset(root=root, train=False, download=True, transform=None)
    meanstd = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    testloader = condensa.data.GPUDataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             meanstd=meanstd)
    return testloader
