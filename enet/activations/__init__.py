from enet.activations.sigmoid import sigmoid, sigmoid_derive
from enet.activations.relu import relu, relu_derive
from enet.activations.softmax import softmax, softmax_derive

activation_dict = {
    "sigmoid": sigmoid,
    "relu": relu,
    "softmax": softmax,
}

derive_dict = {
    "sigmoid": sigmoid_derive,
    "relu": relu_derive,
    "softmax": softmax_derive,
}