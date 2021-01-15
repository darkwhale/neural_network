from enet.loss.mse import Mse
from enet.loss.cross_entropy import SoftMaxCrossEntropy


loss_dict = {
    "mse": Mse,
    "cross_entropy": SoftMaxCrossEntropy
}