# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:59:39 2019

@author: jingzl

KNN MNIST
"""
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
batch_size = 100

#MNIST dataset
train_dataset = dsets.MNIST(root='C:\jingzl\workspace\GITHUB\KNN\ml\pymnist',
                            train = True,
                            transform = None,
                            download = True)
test_dataset = dsets.MNIST( root='C:\jingzl\workspace\GITHUB\KNN\ml\pymnist',
                           train = False,
                           transform = None,
                           download = True)
# 加载数据
train_loader = torch.utils.data.DataLoader( dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader( dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)



