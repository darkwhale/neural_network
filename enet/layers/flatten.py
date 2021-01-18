from enet.layers.base_layer import Layer

import numpy as np


class Flatten(Layer):
    """
    flatten层，将多维的张量展开成1维的
    """

    def __init__(self):
        super(Flatten, self).__init__(layer_type="flatten")

    def build(self, input_shape=None):
        self.input_shape = input_shape
        self.output_shape = np.prod(self.input_shape)

    def forward(self, input_signal, train=False):

        return input_signal.reshape((input_signal.shape[0], -1))

    def backward(self, delta):
        return delta.reshape((-1, ) + self.input_shape)
