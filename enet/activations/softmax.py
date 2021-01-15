import numpy as np


def softmax(input_signal):
    """
    softmax函数
    :param input_signal:
    :return:
    """
    input_signal = input_signal - np.max(input_signal)
    return np.exp(input_signal) / np.sum(np.exp(input_signal), axis=-1, keepdims=True)


def softmax_derive(delta, input_signal):
    """
    softmax导函数
    :param delta:
    :param input_signal:
    :return:
    """
    out_signal = []
    input_signal = softmax(input_signal)

    for sub_signal, sub_delta in zip(input_signal, delta):
        out = np.diag(sub_signal) - np.outer(sub_signal, sub_signal)
        out_signal.append(np.dot(out, sub_delta))

    return np.array(out_signal)
