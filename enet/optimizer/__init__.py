from enet.optimizer.momentum import Momentum
from enet.optimizer.sgd import SGD
from enet.optimizer.adagrad import AdaGrad
from enet.optimizer.rmsprop import RmsProp
from enet.optimizer.adam import Adam

optimizer_dict = {
    "sgd": SGD,
    "momentum": Momentum,
    "adagrad": AdaGrad,
    "rmsprop": RmsProp,
    "adam": Adam
}