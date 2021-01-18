from enet.model import Sequential
from enet.data import ImageHandler
from enet.layers import Dense, Sigmoid, Dropout, Softmax, BatchNormalization, Conv2D, Flatten, MaxPool2D


if __name__ == '__main__':

    data_handler = ImageHandler("/Users/zxy/Documents/meching_learning/neural_network/dataset",
                                gray=True, flatten=False, use_scale=True)
    train_data, train_label, test_data, test_label = data_handler.get_data(ratio=0.2)

    model = Sequential()

    model.add(Conv2D(filters=32, activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=64, activation="relu"))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(kernel_size=64, optimizer="adam"))
    model.add(Sigmoid())
    model.add(Dense(kernel_size=10, activation="sigmoid", optimizer="adam"))

    model.compile(loss="mse", lr=0.01)
    model.summary()

    model.fit(train_data, train_label, epoch=20)
