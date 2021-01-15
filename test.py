from enet.model import Sequential
from enet.data import DataHandler
from enet.layers import Dense, Sigmoid


if __name__ == '__main__':

    data_handler = DataHandler("dataset", gray=True, flatten=True, use_scale=True)
    train_data, train_label, test_data, test_label = data_handler.get_data(ratio=0.2)

    model = Sequential()

    model.add(Dense(input_shape=(784, ), kernel_size=64, activation="sigmoid", optimizer="autograd"))
    # model.add(Dense(kernel_size=64, activation="sigmoid", optimizer="sgd"))
    # model.add(Dense(kernel_size=64, activation="sigmoid", optimizer="sgd"))
    model.add(Dense(kernel_size=32, activation="sigmoid", optimizer="autograd"))
    model.add(Dense(kernel_size=10, activation="sigmoid", optimizer="autograd"))

    model.compile(loss="cross_entropy", lr=0.1)
    model.summary()

    model.fit(train_data, train_label, epoch=20)
