__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


from abc import ABCMeta, abstractmethod
from dies.utils_pytorch import freeze, unfreeze
from fastai.layers import *
from torch import nn


class Unfreeze(nn.Module):
    def unfreeze(self):
        flattend_model = flatten_model(self)
        for l in flattend_model:
            unfreeze(l)


class Freeze(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def freeze_to(self, n, include_embedding=False):
        raise NotImplementedError


class Transfer(Unfreeze, Freeze):
    def __init__(self):
        super().__init__()
