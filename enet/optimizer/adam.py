import numpy as np


class Adam(object):
    """
    momentum优化器
    """

    def __init__(self, beta1=0.9, beta2=0.999, **k_args):

        self.m = dict()
        self.v = dict()
        self.params = dict()

        self.iter = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def grand(self, **k_args):
        """
        记录当前的梯度信息
        :param k_args: 梯度参数
        :return:
        """
        for key, array in k_args.items():
            if key not in self.params:
                self.params[key] = np.zeros_like(array)
                self.m[key] = np.zeros_like(array)
                self.v[key] = np.zeros_like(array)

            self.params[key] += array

    def get_delta_and_reset(self, lr, *args):
        """
        获取obj对象下的梯度
        :param lr: 学习率
        :param args: 需要取出的梯度
        :return:
        """

        self.iter += 1

        lr_t = lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        result_list = []
        for key in args:
            self.m[key] = (1 - self.beta1) * (self.params[key] - self.m[key])
            self.v[key] = (1 - self.beta2) * (self.params[key] ** 2 - self.v[key])

            result = - lr_t * self.m[key] / np.sqrt(self.v[key] + 1e-7)
            result_list.append(result)

            # 更新完后需要将params的梯度置为0
            self.params[key] *= 0

        return result_list
