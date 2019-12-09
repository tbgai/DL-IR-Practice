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
import matplotlib.pyplot as plt

batch_size = 100

# MNIST dataset
# 下载一次即可，然后在本地直接使用
train_dataset = dsets.MNIST(root='C:\jingzl\workspace\GITHUB\DL-IR-Practice\KNN\ml\pymnist',
                            train = True,
                            transform = None,
                            download = False) # download=True
test_dataset = dsets.MNIST( root='C:\jingzl\workspace\GITHUB\DL-IR-Practice\KNN\ml\pymnist',
                           train = False,
                           transform = None,
                           download = False) # download = True
# 加载数据
train_loader = torch.utils.data.DataLoader( dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader( dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)
print("train_data:", train_dataset.train_data.size())
print("train_labels:", train_dataset.train_labels.size())
print("test_data:", test_dataset.test_data.size())
print("test_labels:",test_dataset.test_labels.size())

digit = train_loader.dataset.train_data[0] # 取第一个图片的数据
plt.imshow( digit, cmap=plt.cm.binary )
plt.show()
print( train_loader.dataset.train_labels[0]) # 输出对应的标签



