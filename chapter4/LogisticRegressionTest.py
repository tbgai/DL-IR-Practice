# Sigmoid函数、梯度下降法、
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x) )
    return y

def drawSigmoid():
    plot_x = np.linspace(-10, 10, 100)
    plot_y = sigmoid(plot_x)
    plt.plot(plot_x, plot_y)
    plt.show()


if __name__ == '__main__':
    # draw sigmoid
    #drawSigmoid()

    # 测试梯度下降法
    plot_x = np.linspace(-1, 6, 141)
    plot_y = (plot_x - 2.5) ** 2 - 1  # 二次方程的损失函数
    plt.scatter(plot_x[5], plot_y[5], color='r')
    plt.plot(plot_x, plot_y)
    plt.xlabel('theta', fontproperties='simHei', fontsize=15)
    plt.ylabel('损失函数', fontproperties='simHei', fontsize=15)
    plt.show()




