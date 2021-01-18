from enet.layers.dense import Dense
from enet.layers.base_layer import Layer
from enet.layers.sigmoid import Sigmoid
from enet.layers.relu import Relu
from enet.layers.dropout import Dropout
from enet.layers.softmax import Softmax
from enet.layers.batch_normalization import BatchNormalization
from enet.layers.convolution import Conv2D
from enet.layers.flatten import Flatten
from enet.layers.max_pooling import MaxPool2D


activation_dict = {
    "sigmoid": Sigmoid,
    "relu": Relu,
    "softmax": Softmax
}
