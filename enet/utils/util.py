import numpy as np
from collections import Iterable


def normal_layer_map_shape(shape):
    """
    标准化map形状
    :param shape: map形状
    :return: 标准化后的map形状
    """
    if not isinstance(shape, Iterable):
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


def img2col(image, kernel_size, stride):
    """
    img2col的实现，加速卷积运算
    :param image: 图像
    :param kernel_size: 核大小例如(3, 3)
    :param stride: 步长
    :return: 生成的矩阵, 为3维, batch,
    """
    batch_size, height, width, channel = image.shape
    out_h, out_w = (height - kernel_size[0]) // stride + 1, (width - kernel_size[1]) // stride + 1

    image_col = np.empty(shape=(batch_size, out_h * out_w, kernel_size[0] * kernel_size[1] * channel))
    for i in range(0, out_h, stride):
        h_min = i * stride
        h_max = i * stride + kernel_size[0]
        for j in range(out_w):
            w_min = j * stride
            w_max = j * stride + kernel_size[1]

            image_col[:, i * out_w + j, :] = image[:, h_min: h_max, w_min: w_max, :].reshape((batch_size, -1))

    return image_col


def col2img(image_col, kernel_size, padding_shape, stride):
    """
    col2img的实现，回传梯度
    :param stride: 步长
    :param image_col: 图像列信息
    :param kernel_size: 核大小
    :param padding_shape: 原图像pad之后的图像大小
    :return:
    """
    batch_size, height, width, channel = padding_shape
    out_h, out_w = (height - kernel_size[0]) // stride + 1, (width - kernel_size[1]) // stride + 1

    padding_image = np.zeros(shape=padding_shape)

    for i in range(out_h):
        h_min = i * stride
        h_max = i * stride + kernel_size[0]
        for j in range(out_w):
            w_min = j * stride
            w_max = j * stride + kernel_size[1]

            padding_image[:, h_min: h_max, w_min: w_max, :] += image_col[:, i * out_w + j, :].reshape((batch_size,
                                                                                                       kernel_size[0],
                                                                                                       kernel_size[1],
                                                                                                       channel))
    return padding_image
