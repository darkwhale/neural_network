import numpy as np
import pickle

from enet.loss import loss_dict
from enet.layers import activation_dict
from enet.utils import normal_layer_map_shape, train_test_split, shuffle_data_label
from enet.activations import softmax


class Sequential(object):
    """
    管道模型，用于堆叠网络层
    """

    def __init__(self, layer_list=None, name=None):
        """
        初始化网络模型
        :param layer_list: 网络层列表
        :param name: 模型名称
        """
        self.layer_list = layer_list if layer_list else []

        self.name = name if name else "sequential"

        self.loss = None
        self.optimizer = None

        self.lr = None

    def add(self, layer):
        """
        添加网络层
        :param layer: 神经网络层
        :return: 无返回
        """
        self.layer_list.append(layer)

    def compile(self, loss="mse", lr=0.01, **k_args):
        """
        编译模型
        :param loss: 损失函数
        :param lr: 学习率
        :param k_args: 其他参数， 比如momentum
        :return:
        """

        # 这里只实现两种损失函数
        assert loss in {"mse", "cross_entropy"}

        # self.loss赋值为Mse()或CrossEntropy()
        # self.optimizer赋值为SGD、Momentum...
        self.loss = loss_dict[loss]()

        self.lr = lr

        input_shape = None
        # 开始编译模型
        # 建立新的列表，准备插入激活层等；
        new_layer_list = []
        for index, layer in enumerate(self.layer_list):
            if index == 0:
                input_shape = layer.get_input_shape()

            layer.build(input_shape)

            new_layer_list.append(layer)

            # 下一层输入神经但愿个数等于该层个数
            input_shape = layer.get_output_shape()

            if layer.get_activation_layer():
                new_layer = activation_dict[layer.get_activation_layer()]()
                new_layer.build(input_shape)

                new_layer_list.append(new_layer)

        self.layer_list = new_layer_list

        return self

    def summary(self):
        """
        打印网络形状
        :return:
        """
        print("{} structure: ".format(self.name))

        result_list = []
        current_layer_type = None
        for index, layer in enumerate(self.layer_list):
            if index == 0:
                result_list.append("-" * 60)
                result_list.append("{}{}{}{}".format(
                    "layer type".ljust(15),
                    "input map".ljust(15),
                    "output map".ljust(15),
                    "weight shape".ljust(15)
                ))
                result_list.append("-" * 60)
                result_list.append("{}{}{}{}".format(
                    "input_layer".ljust(15),
                    normal_layer_map_shape(layer.get_input_shape()).ljust(15),
                    normal_layer_map_shape(layer.get_input_shape()).ljust(15),
                    None
                ))
            if current_layer_type != layer.get_layer_type():
                result_list.append("-" * 60)
            result_list.append("{}{}{}{}".format(
                layer.get_layer_type().ljust(15),
                normal_layer_map_shape(layer.get_input_shape()).ljust(15),
                normal_layer_map_shape(layer.get_output_shape()).ljust(15),
                ", ".join(str(param) for param in layer.get_weight_shape()).ljust(15) if layer.get_weight_shape()
                else "None"
            ))

        result_list.append("-" * 60)
        for result in result_list:
            print(result)

    def forward(self, input_data=None, train=True):
        """
        前向运算
        :param train: 是否为训练模式
        :param input_data: 输入数据
        :return: 返回输出
        """
        output_signal = input_data

        for index, layer in enumerate(self.layer_list):
            output_signal = layer.forward(output_signal, train=train)

        return output_signal

    def predict(self, input_data=None):
        """
        预测输出的概率
        :param input_data: 输入数据
        :return:
        """
        # 添加softmax层
        output_signal = self.forward(input_data, train=False)
        return softmax(output_signal)

    def predict_class(self, input_data, class_dict=None):
        """
        输出预测的种类
        :param input_data: 输入数据
        :param class_dict: 标签对应的种类字典，如果为空，则只输出标签
        :return: 预测标签
        """
        output_signal = self.forward(input_data, train=False)

        # 延列方向取最大的索引
        result_array = np.argmax(output_signal, axis=-1)

        if class_dict:
            return [class_dict.get(result, None) for result in result_array]
        else:
            return [result for result in result_array]

    def evaluate(self, input_data, input_label, evaluate_batch=32):
        """
        计算数据的准确率
        :param evaluate_batch: 步长
        :param input_data: 输入数据
        :param input_label: 真是标签
        :return: 准确率
        """

        correct_num = 0
        for index in range(0, len(input_data), evaluate_batch):
            batch_data, batch_label = input_data[index: index + evaluate_batch], \
                                      input_label[index: index + evaluate_batch]

            y_predict_class = self.predict_class(batch_data)
            y_true_class = np.argmax(batch_label, axis=-1)

            correct_num += np.sum(y_predict_class == y_true_class)

        return correct_num / len(input_data)

    def backward(self, delta, lr):
        """
        反向传播
        :param lr: 学习率
        :param delta: 梯度
        :return:
        """
        current_delta = delta
        for layer in self.layer_list[::-1]:
            current_delta = layer.backward(current_delta)
            layer.update(lr)

    def fit(self, train_data=None, train_label=None, val_ratio=0.2, epoch=10, batch=32):
        """
        训练模型
        :param train_data: 训练数据
        :param train_label: 训练标签
        :param val_ratio: 验证集比例
        :param epoch: 迭代代数
        :param batch: 批处理大小
        :return:
        """
        # 划分训练集和验证集
        train_data, train_label, val_data, val_label = train_test_split(train_data, train_label, val_ratio)

        for i in range(epoch):

            # 作梯度更新之前先打乱训练数据的顺序
            train_data, train_label = shuffle_data_label(train_data, train_label)

            for index in range(0, len(train_data), batch):
                batch_data, batch_label = train_data[index: index + batch], train_label[index: index + batch]

                y_predict = self.forward(batch_data, train=True)

                # 计算当前损失
                loss = self.loss.calculate_loss(y_predict, batch_label)

                delta = self.loss.derivative()
                self.backward(delta, self.lr)

                y_predict_class = np.argmax(y_predict, axis=-1)
                y_label_class = np.argmax(batch_label, axis=-1)
                accuracy = np.sum(y_predict_class == y_label_class) / batch

                process_percent = index / len(train_data)
                print("\repoch_{} {} loss:{}\tacc:{}".format(i + 1,
                                                             ("-" * int(100 * process_percent)).ljust(100, " "),
                                                             loss,
                                                             accuracy), end="", flush=True)

            # 输出换行符
            print()

            print(
                "validation data size: {}, accuracy: {}".format(len(val_data), self.evaluate(val_data, val_label)))

    def save(self, file_path):
        """
        保存模型
        :param file_path:
        :return:
        """
        with open(file_path, "wb") as writer:
            pickle.dump(self.layer_list, writer)

    def load_weight(self, file_path):
        """
        载入模型变量
        :param file_path:
        :return:
        """
        with open(file_path, "rb") as reader:
            self.layer_list = pickle.load(reader)
