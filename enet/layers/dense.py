from enet.layers.base_layer import Layer

import numpy as np

from enet.optimizer import optimizer_dict


class Dense(Layer):
    """
    全连接神经网络类
    """

    def __init__(self, kernel_size=None, activation=None, input_shape=None, optimizer="sgd", use_bias=True, **k_args):
        """
        :param kernel_size: 神经元个数
        :param activation: 激活函数
        :param input_shape: 输入shape，只在输入层有效；
        :param optimizer: 优化器；
        :param use_bias: 是否使用偏执，保留参数
        """
        super(Dense, self).__init__()

        assert activation in {None, "sigmoid", "relu", "softmax"}
        assert optimizer in {"sgd", "momentum", "adagrad", "adam", "rmsprop"}

        self.output_shape = kernel_size
        self.activation = activation

        self.layer_type = "dense"

        # 该处的input_shape只在输入层有效,input_shape样式为(784,)
        if input_shape:
            self.input_shape = input_shape[0]

        self.weight = None
        self.bias = None

        # self.use_bias = use_bias
        self.optimizer = optimizer_dict[optimizer](**k_args)

    def build(self, input_shape):
        """
        根据input_shape来构建网络模型参数
        :param input_shape: 输入形状
        :return: 无返回值
        """

        last_dim = input_shape
        self.input_shape = input_shape

        shape = (last_dim, self.output_shape)
        self.weight_shape = shape

        self.weight = self.add_weight(shape=shape, initializer="normal", node_num=input_shape)
        self.bias = self.add_weight(shape=(self.output_shape,), initializer="zero")

    def forward(self, input_signal, *args, **k_args):
        """
        前向传播
        :param input_signal: 输入信息
        :return: 输出信号
        """
        self.cache = input_signal

        return np.dot(input_signal, self.weight) + self.bias

    def backward(self, delta):
        """
        反向传播
        :param delta: 输入梯度
        :return: 误差回传
        """

        # if self.use_bias:
        #     delta_b = np.mean(delta, axis=0)
        # else:
        #     delta_b = 0
        delta_b = np.sum(delta, axis=0)
        delta_w = np.dot(self.cache.transpose(), delta)

        self.optimizer.grand(delta_w=delta_w, delta_b=delta_b)

        # 回传给前一层的梯度
        return np.dot(delta, self.weight.transpose())

    def update(self, lr):
        """
        更新参数
        :param lr: 学习率
        :return:
        """
        delta_w, delta_b = self.optimizer.get_delta_and_reset(lr, "delta_w", "delta_b")

        self.weight += delta_w
        self.bias += delta_b



