import numpy as np


def relu(input_signal):
    """
    relu函数
    :param input_signal: 输入信息
    :return: 输出信息
    """
    # return 1 * (input_signal > 0) * input_signal
    return np.maximum(0., input_signal)


def relu_derive(delta, input_signal):
    """
    relu导函数
    :param delta: 回传的梯度
    :param input_signal: 输入数据
    :return: 输入数据的导数
    """

    return delta * (input_signal > 0).astype(int)
