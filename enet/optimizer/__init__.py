from enet.optimizer.momentum import Momentum
from enet.optimizer.sgd import SGD
from enet.optimizer.autograd import AutoGrad

optimizer_dict = {
    "sgd": SGD,
    "momentum": Momentum,
    "autograd": AutoGrad
}