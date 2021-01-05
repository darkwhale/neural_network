import numpy as np


class Layer(object):
    """
    基础网络层
    """

    def __init__(self):
        pass

    @staticmethod
    def add_weight(shape=None,
                   dtype=np.float,
                   initializer="normal",
                   node_num=None):
        """
        初始化网络参数
        :param node_num: 上一层神经网络节点的数量
        :param shape: 参数的shape
        :param dtype: 参数的dtype
        :param initializer: 初始化方法
        :return: 初始化后的数据
        """

        if initializer == "zero":
            return np.zeros(shape=shape, dtype=dtype)
        if initializer == "normal":
            return np.random.normal(size=shape) * np.sqrt(2 / np.prod(node_num))

        raise TypeError("initializer must be normal or zero")

    def build(self, **k_args):
        pass

    def forward(self, **k_args):
        pass

    def backward(self, **k_args):
        pass

    def get_input_shape(self):
        pass

    def get_output_shape(self):
        pass

    def get_layer_type(self):
        pass
