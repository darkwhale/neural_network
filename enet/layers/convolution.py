from enet.layers.base_layer import Layer
from enet.optimizer import optimizer_dict
from enet.utils import img2col, col2img

import numpy as np


class Conv2D(Layer):
    """
    2维卷积层
    """

    def __init__(self, filters, kernel_size=(3, 3), strides=1, padding="same", activation=None, optimizer="sgd",
                 input_shape=None, use_bias=True, **k_args):
        """
        初始化变量
        :param filters: 卷积核个数
        :param kernel_size: 卷积核大小
        :param strides: 卷积步长
        :param padding: same或valid
        :param activation: 激活函数
        :param optimizer: 优化器
        :param input_shape:
        :param use_bias:
        :param k_args:
        """
        super(Conv2D, self).__init__(layer_type="conv2d")

        assert padding.lower() in {"same", "valid"}

        self.filters = filters

        if input_shape:
            self.input_shape = input_shape

        self.kernel_size = kernel_size
        self.strides = strides

        self.padding = padding

        assert activation in {None, "sigmoid", "relu", "softmax"}
        assert optimizer in {"sgd", "momentum", "adagrad", "adam", "rmsprop"}

        self.activation = activation
        self.optimizer = optimizer_dict[optimizer](**k_args)

        self.weight = None
        self.bias = None

        # padding_shape用于记录pad之后的大小
        self.padding_shape = None
        self.cache_weight = None

    def build(self, input_shape):
        """
        根据input_shape来构建网络模型参数
        :param input_shape: 输入形状
        :return: 无返回值
        """
        self.input_shape = input_shape

        self.weight_shape = (self.kernel_size + (input_shape[-1], self.filters))
        self.weight = self.add_weight(shape=self.weight_shape, node_num=input_shape)
        self.bias = self.add_weight(shape=(self.filters, ), initializer="zero")

        if self.padding == "same":
            self.output_shape = ((input_shape[0] - 1) // self.strides + 1,
                                 (input_shape[1] - 1) // self.strides + 1,
                                 self.filters)
        else:
            self.output_shape = ((input_shape[0] - self.kernel_size[0]) // self.strides + 1,
                                 (input_shape[1] - self.kernel_size[1]) // self.strides + 1,
                                 self.filters)

    def forward(self, input_signal, *args, **k_args):
        """
        前向传播
        :param input_signal: 输入信息
        :param args:
        :param k_args:
        :return:
        """
        # 填充边界数据
        if self.padding == "same":
            input_signal = np.pad(input_signal,
                                  ((0, 0),
                                   (self.kernel_size[0] // 2, self.kernel_size[0] // 2),
                                   (self.kernel_size[1] // 2, self.kernel_size[1] // 2),
                                   (0, 0)),
                                  mode="constant"
                                  )

        self.padding_shape = input_signal.shape

        matrix_weight = self.weight.reshape((-1, self.filters))
        matrix_image = img2col(input_signal, self.kernel_size, self.strides)

        self.cache_weight = matrix_weight
        self.cache = matrix_image

        output_signal = np.matmul(matrix_image, matrix_weight) + self.bias

        return output_signal.reshape((-1,) + self.output_shape)

    def backward(self, delta):
        """
        反向传播
        :param delta: 梯度
        :return:
        """
        delta_col = delta.reshape((delta.shape[0], -1, self.filters))

        delta_w = np.sum(np.matmul(self.cache.transpose(0, 2, 1), delta_col), axis=0).reshape(self.weight.shape)
        delta_b = np.sum(delta_col, axis=(0, 1))

        # 更新到优化器中
        self.optimizer.grand(delta_w=delta_w, delta_b=delta_b)

        delta_padding_image_col = np.matmul(delta_col, self.cache_weight.transpose())
        output_delta = col2img(delta_padding_image_col, self.kernel_size, self.padding_shape, self.strides)

        # 如果padding为same，则需要去除边界
        if self.padding == "same":
            output_delta = output_delta[:,
                                        self.kernel_size[0] // 2: - (self.kernel_size[0] // 2),
                                        self.kernel_size[1] // 2: - (self.kernel_size[1] // 2),
                                        :]

        return output_delta

    def alternative_backward(self, delta):
        """
        另一种反向传播方式
        :param delta: 梯度
        :return:
        """
        delta_col = delta.reshape((delta.shape[0], -1, self.filters))

        delta_w = np.sum(np.matmul(self.cache.transpose(0, 2, 1), delta_col), axis=0).reshape(self.weight.shape)
        delta_b = np.sum(delta_col, axis=(0, 1))

        # 更新到优化器中
        self.optimizer.grand(delta_w=delta_w, delta_b=delta_b)

        # 该方式将传回来的梯度和权值矩阵的翻转结果作卷积运算
        # 先填充delta, 若步长不为1，则需要将回传的梯度填充大小为步长为1的输出大小，其余位置填充0
        if self.padding == "same":
            back_per_stride_height, back_per_stride_width = self.input_shape[0], self.input_shape[1]
        else:
            back_per_stride_height, back_per_stride_width = self.input_shape[0] - self.kernel_size[0] + 1, \
                                                            self.input_shape[1] - self.kernel_size[1] + 1

        if self.strides != 1:
            new_delta = np.zeros(shape=(delta.shape[0],
                                        back_per_stride_height,
                                        back_per_stride_width,
                                        delta.shape[-1]))
            new_delta[:, ::self.strides, ::self.strides, :] = delta
            delta = new_delta

        # weight, 然后输入通道和输出通道变换位置
        flip_weight = np.flip(self.weight, axis=(0, 1)).swapaxes(2, 3).reshape((-1, self.input_shape[-1]))

        # 梯度边界填充
        pixel = [k_size // 2 if self.padding == "same" else k_size - 1 for k_size in self.kernel_size]
        delta = np.pad(delta, ((0, 0), (pixel[0], pixel[0]), (pixel[1], pixel[1]), (0, 0)), mode="constant")
        matrix_delta = img2col(delta, self.kernel_size, 1)

        return np.dot(matrix_delta, flip_weight).reshape((delta.shape[0],) + self.input_shape)

    def update(self, lr):
        """
        更新参数
        :param lr: 学习率
        :return:
        """
        delta_w, delta_b = self.optimizer.get_delta_and_reset(lr, "delta_w", "delta_b")

        self.weight += delta_w
        self.bias += delta_b

