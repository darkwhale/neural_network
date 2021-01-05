import numpy as np


def sigmoid(input_signal):
    """
    sigmoid函数
    :param input_signal: 输入信息
    :return: 输出信息
    """
    return 1. / (1 + np.exp(-input_signal))


def sigmoid_derive(delta, input_signal):
    """
    sigmoid导函数
    :param delta: 回传的梯度
    :param input_signal: 输入数据
    :return: 输入数据的导数
    """
    return delta * sigmoid(input_signal) * (1 - sigmoid(input_signal))
