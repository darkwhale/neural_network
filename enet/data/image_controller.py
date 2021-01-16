import os
import cv2
import numpy as np
import pickle

from enet.utils.util import train_test_split


class ImageHandler(object):
    """
    数据集载入控制类，传入参数为目录，并且目录满足以下特征：
    1. 目录下所有文件都是两级结构，比如 class1/file1.png
    2. 所有文件都为图片类型，可被计算机读取
    3. 同一类图片需要放在同一个目录下

    在生成数据的同时会解析标签字典，其过程为：
    1. 将1级目录排序
    2. 按照顺序生成标签，从0开始
    3. 记录标签到目录名的字典返回class_dict
    """

    def __init__(self, data_dir, gray=False, use_scale=False, flatten=False):
        """
        :param data_dir: 根目录
        :param gray: 是否以灰度化格式读入图片
        :param use_scale: 是否/255.
        :param flatten: 数据结果是否需要拉伸为1维
        """
        self.data_dir = data_dir
        self.gray = gray
        self.use_scale = use_scale
        self.flatten = flatten

        self.class_dict = dict()

    def get_data(self, ratio=0.3, read_cache=True):
        """
        读取数据
        :param ratio: 数据拆分比例
        :param read_cache: 是否使用cache读入，若使用，则直接使用缓存数据，上述参数可能无效
        :return:
        """

        # 这里引入pickle加速读取
        if read_cache:
            if os.path.exists(os.path.join(self.data_dir, "data_cache.pkl")):
                with open(os.path.join(self.data_dir, "data_cache.pkl"), "rb") as reader:
                    return pickle.load(reader)

        train_data, train_label = self.load_data()
        result = train_test_split(train_data, train_label, ratio)

        # 保存到cache文件中
        with open(os.path.join(self.data_dir, "data_cache.pkl"), "wb") as writer:
            pickle.dump(result, writer)

        return result

    def load_data(self):
        """加载图片数据， 返回数据和标签"""
        sub_dir_list = [sub_dir for sub_dir in os.listdir(self.data_dir) if
                        os.path.isdir(os.path.join(self.data_dir, sub_dir))]
        sub_dir_list.sort()

        train_data = []
        train_label = []
        # 遍历文件夹
        for dir_index, sub_dir in enumerate(sub_dir_list):

            for sub_file in os.listdir(os.path.join(self.data_dir, sub_dir)):
                if self.gray:
                    image = cv2.imread(os.path.join(self.data_dir, sub_dir, sub_file), cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(os.path.join(self.data_dir, sub_dir, sub_file))

                image = np.array(image)

                # 灰度模式读取为2维数据，需要添加1维通道信息
                if self.gray:
                    image = np.expand_dims(image, axis=-1)

                # 如果使用flatten，则应该拉成向量
                if self.flatten:
                    image = image.flatten()

                if self.use_scale:
                    image = image / 255.

                # 插入到结果集中
                train_data.append(image)
                train_label.append(dir_index)

        train_label = np.eye(len(sub_dir_list))[train_label]

        return train_data, train_label

    def get_class_dict(self):
        """
        获取类别字典
        :return:
        """
        # 读取数据，去掉缓存文件
        sub_dir_list = [sub_dir for sub_dir in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir,
                                                                                                       sub_dir))]
        sub_dir_list.sort()

        for index, sub_dir in enumerate(sub_dir_list):
            self.class_dict[index] = sub_dir

        return self.class_dict
