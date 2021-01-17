from enet.layers.dense import Dense
from enet.layers.base_layer import Layer
from enet.layers.sigmoid import Sigmoid
from enet.layers.relu import Relu
from enet.layers.dropout import Dropout
from enet.layers.softmax import Softmax
from enet.layers.batch_normalization import BatchNormalization


activation_dict = {
    "sigmoid": Sigmoid,
    "relu": Relu,
    "softmax": Softmax
}
