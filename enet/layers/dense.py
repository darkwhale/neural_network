from enet.layers import Layer
from enet.activations import activation_dict, derive_dict

import numpy as np


class Dense(Layer):
    """
    全连接神经网络类
    """

    def __init__(self, kernel_size=None, activation=None, use_bias=True, input_shape=None):
        """
        :param kernel_size: 神经元个数
        :param activation: 激活函数
        :param use_bias: 是否使用偏执
        :param input_shape: 输入shape，只在输入层有效；
        """
        super(Dense, self).__init__()

        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias

        self.layer_type = "dense"

        # 该处的input_shape只在输入层有效,input_shape样式为(784,)
        if input_shape:
            self.input_shape = input_shape[0]

        self.kernel = None
        self.bias = None
        self.optimizer = None

        self.signal_z = None
        self.momentum = None

        self.last_delta_w = None
        self.last_delta_b = None

        if self.activation:
            self.activation = self.activation.lower()
            assert self.activation in {"sigmoid", "relu", "softmax"}

            self.signal_a = None

            self.activation_method = activation_dict[self.activation]
            self.derive_method = derive_dict[self.activation]

    def build(self, input_shape, optimizer="sgd",  momentum=0.9, **k_args):
        """
        根据input_shape来构建网络模型参数
        :param momentum: 冲量值，适用与optimizer为momentum
        :param optimizer: 优化器
        :param input_shape: 输入形状
        :return: 无返回值
        """

        assert optimizer.lower() in {"sgd", "momentum"}

        last_dim = input_shape
        self.input_shape = input_shape

        self.optimizer = optimizer.lower()
        self.momentum = momentum

        shape = (last_dim, self.kernel_size)

        self.kernel = self.add_weight(shape=shape, initializer="normal")

        if self.optimizer == "momentum":
            self.last_delta_w = np.zeros(shape=shape)
            self.last_delta_b = np.zeros(shape=(self.kernel_size,))

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.kernel_size,), initializer="zero")

    def forward(self, input_signal, train=False):
        """
        前向传播
        :param train: 是否为训练模式
        :param input_signal: 输入信息
        :return: 输出信号
        """
        if train:
            self.signal_z = input_signal

        output_signal = np.dot(input_signal, self.kernel) + self.bias

        if self.activation:
            self.signal_a = output_signal
            output_signal = self.activation_method(output_signal)

        return output_signal

    def backward(self, delta, lr=0.01):
        """
        反向传播
        :param lr: 学习率
        :param delta: 输入梯度
        :return: 误差回传
        """

        # 若使用了激活函数
        if self.activation:
            delta = self.derive_method(delta, self.signal_a)

        # delta_b按照行方向求均值，delta_w除以样本数即可
        # 此时样本数为 input_shape.shape[0]
        delta_b = np.mean(delta, axis=0) * lr
        delta_w = np.dot(self.signal_z.transpose(), delta) / delta.shape[0] * lr

        if self.optimizer == "momentum":
            delta_b += self.momentum * self.last_delta_b
            delta_w += self.momentum * self.last_delta_w

            self.last_delta_b = delta_b
            self.last_delta_w = delta_w

        # 回传给前一层的梯度
        output_delta = np.dot(delta, self.kernel.transpose())

        # 更新权重和偏置
        self.kernel -= delta_w
        self.bias -= delta_b

        # 继续回传梯度
        return output_delta

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.kernel_size

    def get_layer_type(self):
        return self.layer_type


