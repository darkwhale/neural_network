from enet.activations.softmax import softmax

import numpy as np


class SoftMaxCrossEntropy(object):
    """
    softmax cross_entropy函数
    """
    def __init__(self):

        self.cache = None
        self.y_true = None

    def calculate_loss(self, y_hat, y_true):
        """
        计算损失
        :param y_hat: 预测的结果
        :param y_true: 真实结果
        :return:
        """
        self.cache = softmax(y_hat)
        self.y_true = y_true

        return -np.sum((y_true * np.log(self.cache))) / self.y_true.shape[0]

    def derivative(self):
        """
        计算损失梯度
        :return:
        """
        return (self.cache - self.y_true) / self.cache.shape[0]
