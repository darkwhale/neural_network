from enet.layers.dense import Dense
from enet.layers.base_layer import Layer
from enet.layers.sigmoid import Sigmoid
from enet.layers.relu import Relu
from enet.layers.dropout import Dropout


activation_dict = {
    "sigmoid": Sigmoid,
    "relu": Relu
}
