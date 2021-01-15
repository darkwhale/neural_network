import numpy as np


class Momentum(object):
    """
    momentum优化器
    """

    def __init__(self, momentum=0.9, **k_args):

        self.momentum = momentum

        # 相比于一般的实现，我们设立了两个参数，params主要用于累计梯度；
        self.v = dict()
        self.params = dict()

    def grand(self, **k_args):
        """
        记录当前的梯度信息
        :param k_args: 梯度参数
        :return:
        """
        for key, array in k_args.items():
            if key not in self.params:
                self.params[key] = np.zeros_like(array)
                self.v[key] = np.zeros_like(array)

            self.params[key] += array

    def get_delta_and_reset(self, lr, *args):
        """
        获取obj对象下的梯度
        :param lr: 学习率
        :param args: 需要取出的梯度
        :return:
        """
        result_list = []
        for key in args:
            self.v[key] = self.momentum * self.v[key] - lr * self.params[key]
            result_list.append(self.v[key])

            # 更新完后需要将params的梯度置为0
            self.params[key] *= 0

        return result_list
