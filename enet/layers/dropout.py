from enet.layers.base_layer import Layer

import numpy as np


class Dropout(Layer):
    """
    dropout
    """

    def __init__(self, dropout_ratio=0.5):
        """
        dropout层
        :param dropout_ratio:
        """
        super(Dropout, self).__init__()

        self.dropout_ratio = dropout_ratio
        self.mask = None

        self.layer_type = "dropout"

    def build(self, input_shape):
        """
        构建dropout层，此层无参数
        :param input_shape: 输入数据
        :return: 无
        """
        self.input_shape = input_shape
        self.output_shape = input_shape

    def forward(self, input_signal, train=False, *args, **k_args):
        """
        前向运算
        :param train: 是否为训练模式
        :param input_signal: 输入数据
        :return: 输出数据
        """
        if train:
            self.mask = np.random.random(input_signal.shape) > self.dropout_ratio
            return input_signal * self.mask
        else:
            return input_signal * (1 - self.dropout_ratio)

    def backward(self, delta):
        """
        反向传播
        :param delta: 梯度
        :return: 继续向前传播梯度
        """
        return delta * self.mask

