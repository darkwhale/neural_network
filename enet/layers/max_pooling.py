from enet.layers.base_layer import Layer
from enet.utils import img2col, col2img, get_ndim_except_last
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

    def forward(self, input_signal, train=False):

        image_col = img2col(input_signal, (self.size, self.size), self.size)
        image_col = image_col.reshape((image_col.shape[0],
                                       image_col.shape[1],
                                       self.size * self.size, self.input_shape[-1]))

        # 此时导数第二维表示图像块元素，取最大值；
        out_image_col = np.max(image_col, axis=-2)
        self.mask = image_col == out_image_col.repeat(repeats=self.size * self.size, axis=-1)

        return out_image_col.reshape((-1, ) + self.output_shape)

    def backward(self, delta):
        delta_col = np.zeros(shape=(delta.shape[0],
                                    self.output_shape[0] * self.output_shape[1],
                                    self.size * self.size,
                                    self.output_shape[-1]))

        delta_col[:, :, self.mask.flatten(), :] = delta.reshape((delta.shape[0], -1, self.output_shape[-1]))

        delta_image = col2img(delta_col.reshape((delta_col.shape[0], delta_col.shape[1], -1)),
                              (self.size, self.size),
                              (delta.shape[0], ) + self.input_shape,
                              self.size)

        return delta_image
