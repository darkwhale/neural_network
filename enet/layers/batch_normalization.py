from enet.layers.base_layer import Layer
from enet.utils.util import get_ndim_except_last
from enet.optimizer import optimizer_dict

import numpy as np


class BatchNormalization(Layer):
    """
    全连接神经网络类
    """

    def __init__(self, decay=0.9, optimizer="sgd", **k_args):
        """
        :param momentum: 计算全局均值标准差时的冲量
        """
        super(BatchNormalization, self).__init__()

        self.layer_type = "batch_normalization"

        self.decay = decay

        self.running_mean = None
        self.running_var = None

        self.gamma = None
        self.beta = None

        self.param_shape = None
        self.cache_std = None
        self.cache_xc = None
        self.cache_xn = None

        self.optimizer = optimizer_dict[optimizer](**k_args)

    def build(self, input_shape):
        """
        根据input_shape来构建网络模型参数
        :param input_shape: 输入形状
        :return: 无返回值
        """

        self.input_shape = input_shape
        self.output_shape = input_shape

        # 判断input为1维还是多维
        self.param_shape = input_shape if isinstance(input_shape, int) else input_shape[-1]

        self.gamma = np.random.uniform(low=0.9, high=1.1, size=self.param_shape)
        self.beta = np.random.uniform(low=-0.1, high=0.1, size=self.param_shape)

        self.running_mean = np.zeros(shape=self.param_shape)
        self.running_var = np.zeros(shape=self.param_shape)

    def forward(self, input_signal, train, *args, **k_args):
        """
        前向传播
        :param train: 是否维训练模式
        :param input_signal: 输入信息
        :return: 输出信号
        """
        if train:
            sample_mean = np.mean(input_signal, axis=get_ndim_except_last(input_signal))
            sample_var = np.var(input_signal, axis=get_ndim_except_last(input_signal))

            # 保存中间值，用作后续的梯度更新
            self.cache = input_signal
            self.cache_std = np.sqrt(sample_var + 1e-7)
            self.cache_xc = input_signal - sample_mean
            self.cache_xn = self.cache_xc / self.cache_std

            # 注意和momentum优化器的区别
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * sample_mean
            self.running_var = self.decay * self.running_var + (1 - self.decay) * sample_var

            input_signal = self.cache_xn
        else:
            input_signal = (input_signal - self.running_mean) / np.sqrt(self.running_var + 1e-7)

        return input_signal * self.gamma + self.beta

    def backward(self, delta):
        """
        反向传播
        :param delta: 输入梯度
        :return: 误差回传
        """
        delta_gamma = np.sum(self.cache * delta, axis=get_ndim_except_last(delta))
        delta_beta = np.sum(delta, axis=get_ndim_except_last(delta))

        self.optimizer.grand(delta_gamma=delta_gamma, delta_beta=delta_beta)

        # 计算返回前一层的梯度
        # x分为3条线，x->xc, x->mean, x->std;
        xn_delta = delta * self.gamma
        xc_delta = xn_delta / self.cache_std
        std_delta = - np.sum((xn_delta * self.cache_xc) / (self.cache_std * self.cache_std),
                             axis=get_ndim_except_last(delta))
        var_delta = 0.5 * std_delta / self.cache_std
        xc_delta += 2.0 * self.cache_xc * var_delta / np.prod(delta.shape[: -1])

        return xc_delta - np.mean(xc_delta, axis=get_ndim_except_last(delta))

    def update(self, lr):
        """
        更新参数
        :param lr: 学习率
        :return:
        """
        delta_gamma, delta_beta = self.optimizer.get_delta_and_reset(lr, "delta_gamma", "delta_beta")

        self.gamma += delta_gamma
        self.beta += delta_beta



