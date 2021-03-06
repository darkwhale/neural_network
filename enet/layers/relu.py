from enet.layers.base_layer import Layer

import numpy as np


class Relu(Layer):
    """
    relu
    """

    def __init__(self):
        super(Relu, self).__init__(layer_type="relu")

    def build(self, input_shape):
        """
        构建relu层，此层无参数
        :param input_shape: 输入数据
        :return: 无
        """
        self.input_shape = input_shape
        self.output_shape = input_shape

    def forward(self, input_signal, *args, **k_args):
        """
        前向运算
        :param input_signal: 输入数据
        :return: 输出数据
        """
        self.cache = input_signal

        return np.maximum(0., input_signal)

    def backward(self, delta):
        """
        反向传播
        :param delta: 梯度
        :return: 继续向前传播梯度
        """
        return delta * (self.cache > 0).astype(int)

