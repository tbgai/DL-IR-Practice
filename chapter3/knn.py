import operator
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

batch_size = 100


def createDataset():
    group = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5], [1.1, 1.0], [0.5, 1.5]])
    labels = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    return group, labels


def kNN_classify(k, dis, X_train, x_train, Y_test):
    # KNN 分类器
    assert dis == 'E' or dis == 'M', 'dis must E or M, E为欧式距离，M为曼哈顿距离'
    num_test = Y_test.shape[0]
    labellist = []

    if (dis == 'E'):  # 欧式距离
        for i in range(num_test):
            distances = np.sqrt(np.sum(((X_train - np.tile(Y_test[i], (X_train.shape[0], 1))) ** 2), axis=1))
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for i in topK:
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)

    if (dis == 'M'):  # 曼哈顿距离
        for i in range(num_test):
            distances = np.sum(np.abs(X_train - np.tile(Y_test[i], (X_train.shape[0], 1))), axis=1)
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for i in topK:
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)


def getXmean(X_train):
    #X_train = np.reshape(X_train, (X_train.reshape[0], -1))  # 将图片从二维展开为一维
    X_train = X_train.flatten()
    mean_image = np.mean(X_train, axis=0)
    return mean_image


def centralized(X_test, mean_image):
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_test = X_test.astype(np.float)
    X_test -= mean_image  # 减去均值图像，实现零均值化
    return X_test


def testMNIST():
    # MNIST dataset
    # 下载一次即可，然后在本地直接使用 D:\workspace\GITHub\jcobra\DL-IR-Practice\chapter3\ml\pymnist
    train_dataset = dsets.MNIST(root='D:\workspace\GITHub\jcobra\DL-IR-Practice\chapter3\ml\pymnist',
                                train=True,
                                transform=None,
                                download=False)  # download=True
    test_dataset = dsets.MNIST(root='D:\workspace\GITHub\jcobra\DL-IR-Practice\chapter3\ml\pymnist',
                               train=False,
                               transform=None,
                               download=False)  # download = True
    # 加载数据
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    '''
    print("train_data:", train_dataset.train_data.size())
    print("train_labels:", train_dataset.train_labels.size())
    print("test_data:", test_dataset.test_data.size())
    print("test_labels:", test_dataset.test_labels.size())

    digit = train_loader.dataset.train_data[0]  # 取第一个图片的数据
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    print(train_loader.dataset.train_labels[0])  # 输出对应的标签
    '''

    # 验证KNN在MNIST上的效果
    X_train = train_loader.dataset.train_data.numpy()
    # 归一化处理
    mean_image = getXmean(X_train)
    X_train = centralized(X_train, mean_image)
    X_train = X_train.reshape(X_train.shape[0], 28 * 28)
    y_train = train_loader.dataset.train_labels.numpy()
    X_test = test_loader.dataset.test_data[:1000].numpy()
    X_test = centralized(X_test, mean_image)
    X_test = X_test.reshape(X_test.shape[0], 28 * 28)
    y_test = test_loader.dataset.test_labels[:1000].numpy()
    num_test = y_test.shape[0]
    y_test_pred = kNN_classify(5, 'M', X_train, y_train, X_test)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print("Got %d / %d correct => accuracy : %f " % (num_correct, num_test, accuracy))


def testCifar10():
    #Cifar10 dataset
    train_dataset = dsets.CIFAR10(root='D:\workspace\GITHub\jcobra\DL-IR-Practice\chapter3\ml\pycifar',
                                  train=True,
                                  download=False)
    test_dataset = dsets.CIFAR10(root='D:\workspace\GITHub\jcobra\DL-IR-Practice\chapter3\ml\pycifar',
                                  train=False,
                                  download=False)
    # 加载数据
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True)
    # 查看
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    # AttributeError: 'CIFAR10' object has no attribute 'train_data'，修改为data
    digit = train_loader.dataset.data[0]
    plt.imshow( digit, cmap=plt.cm.binary)
    plt.show()
    # AttributeError: 'CIFAR10' object has no attribute 'label_names'
    #print( "{0}".format(train_dataset.data.size))
    #print(classes[train_loader.dataset.label_names[0]])



if __name__ == '__main__':
    # group, labels = createDataset()
    # print( labels=='A' )
    # print( group[labels=='A',0] )
    # plt.scatter(group[labels == 'A', 0], group[labels == 'A', 1], color='r', marker='*')
    # plt.scatter(group[labels == 'B', 0], group[labels == 'B', 1], color='g', marker='+')
    # plt.show()

    # 测试
    # y_test_pred = kNN_classify( 1, 'E', group, labels, np.array([[1.0,2.1],[0.4,2.0]]))
    # print( y_test_pred )

    # test MNIST
    #testMNIST()

    # test Cifar10
    testCifar10()
