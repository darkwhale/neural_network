import numpy as np


def normal_layer_map_shape(shape):
    """
    标准化map形状
    :param shape: map形状
    :return: 标准化后的map形状
    """
    if isinstance(shape, int):
        return "{}, {}".format("None", shape)
    else:
        return "{}, {}".format("None", ", ".join([str(ele) for ele in shape]))


def train_test_split(train_data, train_label, ratio=0.3):
    """
    划分训练机和测试机
    :param train_data:
    :param train_label:
    :param ratio:
    :return:
    """
    train_data, train_label = shuffle_data_label(train_data, train_label)

    # 返回train_data, train_label, test_data, test_label
    test_index = int(len(train_data) * ratio)
    return train_data[test_index:], train_label[test_index:], train_data[: test_index], train_label[: test_index]


def shuffle_data_label(data, label):
    """
    随机化data和label
    :param data:
    :param label:
    :return:
    """
    assert len(data) == len(label)

    data, label = np.array(data), np.array(label)
    shuffle_index = np.random.permutation(len(data))

    return data[shuffle_index], label[shuffle_index]


def get_ndim_except_last(input_array):
    """
    获取input_array的维度序列，除了最后一个维度
    比如input_array为4维，则输出(0, 1, 2)
    :param input_array: 输入numpy数组
    :return:
    """
    ndim = input_array.ndim
    return tuple(range(ndim)[: -1])
