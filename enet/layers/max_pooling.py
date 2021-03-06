from enet.layers.base_layer import Layer
from enet.utils import tailor_border, add_border
import numpy as np


class MaxPool2D(Layer):

    def __init__(self, size=2):
        super(MaxPool2D, self).__init__("max_pooling_2d")
        self.size = size
        self.mask = None

    def build(self, input_shape=None):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // self.size,
                             input_shape[1] // self.size,
                             input_shape[-1])

    def forward(self, input_signal, *args, **k_args):

        # 截取多余的边界
        input_signal = tailor_border(input_signal, self.size)

        input_signal = input_signal.reshape(input_signal.shape[0],
                                            input_signal.shape[1] // self.size,
                                            self.size,
                                            input_signal.shape[2] // self.size,
                                            self.size,
                                            input_signal.shape[3]
                                            )
        output_signal = np.max(input_signal, axis=(2, 4))
        self.mask = output_signal.repeat(repeats=self.size, axis=1).repeat(repeats=self.size, axis=2) != input_signal

        return output_signal

    def backward(self, delta):
        delta = delta.repeat(repeats=self.size, axis=1).repeat(repeats=self.size, axis=2)

        delta[self.mask] = 0.

        return add_border(delta, self.input_shape, self.size)

