import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def init_network():
    network = {}
    weight_scale = 1e-3
    network['W1'] = np.random.randn(784, 50) * weight_scale
    network['b1'] = np.ones(50)
    network['W2'] = np.random.randn(50, 100) * weight_scale
    network['b2'] = np.ones(100)
    network['W3'] = np.random.randn(100, 10) * weight_scale
    network['b3'] = np.ones(10)
    return network


def _relu(x):
    return np.maximum(0, x)


def NForword(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = x.dot(w1) + b1
    z1 = _relu(a1)
    a2 = z1.dot(w2) + b2
    z2 = _relu(a2)
    a3 = z2.dot(w3) + b3
    y = a3
    return y


if __name__ == '__main__':
    # MNIST dataset
    # 下载一次即可，然后在本地直接使用 D:\workspace\GITHub\jcobra\DL-IR-Practice\ml\pymnist
    train_dataset = dsets.MNIST(root='D:\workspace\GITHub\jcobra\DL-IR-Practice\ml\pymnist',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=False)  # download=True
    test_dataset = dsets.MNIST(root='D:\workspace\GITHub\jcobra\DL-IR-Practice\ml\pymnist',
                               train=False,
                               transform=transforms.ToTensor(),
                               download=False)  # download = True

    network = init_network()
    accuracy_cnt = 0
    x = test_dataset.test_data.numpy().reshape(-1, 28*28)
    labels = test_dataset.test_labels.numpy()  # tensor 转 numpy
    for i in range(len(x)):
        y = NForword(network, x[i])
        p = np.argmax(y)  # 获取概率最高的元素的索引
        if p == labels[i]:
            accuracy_cnt += 1
    print("Accuracy: " + str(float(accuracy_cnt) / len(x) * 100) + "%")




