import tensorflow.keras as keras
import os
import cv2


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    for i in range(10):
        if not os.path.exists(os.path.join("dataset", str(i))):
            os.mkdir(os.path.join("dataset", str(i)))

    index_list = [0] * 10

    # x_train为图片，y_train为标签；
    for i in range(x_train.shape[0]):
        file_name = os.path.join("dataset", str(y_train[i]), str(index_list[y_train[i]]) + '.bmp')
        index_list[y_train[i]] += 1
        cv2.imwrite(file_name, x_train[i])

    for i in range(x_test.shape[0]):
        file_name = os.path.join("dataset", str(y_test[i]), str(index_list[y_test[i]]) + '.bmp')
        index_list[y_test[i]] += 1
        cv2.imwrite(file_name, x_test[i])
