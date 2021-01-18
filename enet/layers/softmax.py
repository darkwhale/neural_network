from enet.layers.base_layer import Layer
from enet.activations import softmax

import numpy as np


class Softmax(Layer):
    """
    softmax层
    """

    def __init__(self):
        super(Softmax, self).__init__(layer_type="softmax")

    def build(self, input_shape):
        """
        softmax，此层无参数
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
        self.cache = softmax(input_signal)

        return self.cache

    def backward(self, delta):
        """
        反向传播
        :param delta: 梯度
        :return: 继续向前传播梯度
        """
        output_delta = self.cache * delta
        delta_sum = np.sum(output_delta, axis=-1, keepdims=True)
        output_delta -= self.cache * delta_sum
        return output_delta

