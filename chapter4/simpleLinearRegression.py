# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:20:03 2020

@author: jingzl

一元线性回归算法
"""

import numpy as np
import matplotlib.pyplot as plt
from LinearRegressionSelf import SimpleLinearRegressionSelf


def test():
    x = np.array([1, 2, 4, 6, 8])
    y = np.array([2, 5, 7, 8, 9])
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denominator = 0.0
    numerator = 0.0
    for x_i, y_i in zip(x, y):
        numerator += (x_i - x_mean) * (y_i - y_mean)
        denominator += (x_i - x_mean) ** 2
    a = numerator / denominator
    b = y_mean - a * x_mean
    y_predict = a * x + b
    plt.scatter(x, y, color='b')
    plt.plot(x, y_predict, color='r')
    plt.xlabel('管子的长度', fontproperties='simHei', fontsize=15)
    plt.ylabel('收费', fontproperties='simHei', fontsize=15)
    plt.show()
    # 测试
    x_test = 7
    y_predict_value = a * x_test + b
    print(y_predict_value)


if __name__ == '__main__':
    # test()

    # 测试一元线性回归类
    x = np.array([1, 2, 4, 6, 8])
    y = np.array([2, 5, 7, 8, 9])
    lr = SimpleLinearRegressionSelf()
    lr.fit(x, y)
    print(lr.predict([7]))
    print(lr.r_square([8, 9], lr.predict([6, 8])))
