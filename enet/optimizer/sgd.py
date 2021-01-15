import numpy as np


class SGD(object):
    """
    momentum优化器
    """

    def __init__(self, **k_args):

        # params主要用于有分枝的网络连接，比如有两个分支，则需要先汇总梯度，然后再作为总体计算；
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

            self.params[key] += array

    def get_delta_and_reset(self, lr, *args):
        """
        获取obj对象下的梯度
        :param lr: 学习率
        :param args: 参数列表
        :return:
        """
        result_list = []
        for key in args:
            result_list.append(- lr * self.params[key])

            # 更新完后需要将params的梯度置为0
            self.params[key] *= 0

        return result_list
