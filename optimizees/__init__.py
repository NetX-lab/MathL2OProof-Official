from .base import BaseOptimizee
from .lasso_lista import LASSO_LISTA
from .logistic_l1 import LogisticL1
from .qp import QP
from .cnn import MnistCNN
# from .logistic_l1_cifar10 import LogisticL1CIFAR10

OPTIMIZEE_DICT = {
    'QuadraticUnconstrained': QP,
    # 'LASSO': LASSO,
    'LogisticL1': LogisticL1,
    'LASSO_LISTA': LASSO_LISTA,
    'MnistCNN': MnistCNN,
    # 'LogisticL1CIFAR10': LogisticL1CIFAR10,
}

