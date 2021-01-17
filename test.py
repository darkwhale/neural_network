from enet.model import Sequential
from enet.data import ImageHandler
from enet.layers import Dense, Sigmoid, Dropout, Softmax, BatchNormalization


if __name__ == '__main__':

    data_handler = ImageHandler("dataset", gray=True, flatten=True, use_scale=True)
    train_data, train_label, test_data, test_label = data_handler.get_data(ratio=0.2)

    model = Sequential()

    model.add(Dense(input_shape=(784, ), kernel_size=64, optimizer="adam"))
    model.add(BatchNormalization())
    model.add(Sigmoid())
    # model.add(Dense(kernel_size=64, activation="sigmoid", optimizer="sgd"))
    # model.add(Dense(kernel_size=64, activation="sigmoid", optimizer="sgd"))
    model.add(Dense(kernel_size=32, activation="sigmoid", optimizer="adam"))
    model.add(BatchNormalization())
    model.add(Dense(kernel_size=10, activation="sigmoid", optimizer="adam"))
    # model.add(Softmax())

    model.compile(loss="mse", lr=0.01)
    model.summary()

    model.fit(train_data, train_label, epoch=20)
