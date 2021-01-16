import numpy as np


class RmsProp(object):
    """
    momentum优化器
    """

    def __init__(self, decay_rate=0.99, **k_args):

        self.h = dict()
        self.params = dict()

        self.decay_rate = decay_rate

    def grand(self, **k_args):
        """
        记录当前的梯度信息
        :param k_args: 梯度参数
        :return:
        """
        for key, array in k_args.items():
            if key not in self.params:
                self.params[key] = np.zeros_like(array)
                self.h[key] = np.zeros_like(array)

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
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * self.params[key] * self.params[key]
            result = - lr * self.params[key] / (np.sqrt(self.h[key]) + 1e-7)
            result_list.append(result)

            # 更新完后需要将params的梯度置为0
            self.params[key] *= 0

        return result_list
