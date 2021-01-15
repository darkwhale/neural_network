import numpy as np


class Layer(object):
    """
    基础网络层
    """

    def __init__(self):

        self.input_shape = None
        self.output_shape = None
        self.weight_shape = None
        self.layer_type = None
        self.activation = None

        self.cache = None

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
            return np.random.normal(size=shape) * np.sqrt(1 / np.prod(node_num))
            # return np.random.normal(size=shape)

        raise TypeError("initializer must be normal or zero")

    def build(self, *args, **k_args):
        """
        编译网络层
        :param k_args:
        :return:
        """
        pass

    def forward(self, *args, **k_args):
        """
        前向运算
        :param k_args:
        :return:
        """
        pass

    def backward(self, *args, **k_args):
        """
        反向传播，只计算梯度而不更新参数
        :param k_args:
        :return:
        """
        pass

    def update(self, *args, **k_args):
        """
        更新参数
        :param k_args:
        :return:
        """
        pass

    def get_input_shape(self):
        """
        获取网络输入形状
        :return:
        """
        return self.input_shape

    def get_output_shape(self):
        """
        获取网络输出形状
        :return:
        """
        return self.output_shape

    def get_layer_type(self):
        """
        获取网络类型
        :return:
        """
        return self.layer_type

    def get_activation_layer(self):
        """
        获取激活函数类型
        :return:
        """
        return self.activation

    def get_weight_shape(self):
        """
        获取权重形状
        :return:
        """
        return self.weight_shape
