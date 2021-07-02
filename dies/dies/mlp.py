__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


import copy

import torch
import torch.nn as nn

from torch.nn import ReLU

from fastai.tabular.model import *
from fastai.vision.all import *

from blitz.utils import variational_estimator

from dies.utils_pytorch import freeze, unfreeze
from dies.abstracts import Transfer
from dies.utils_blitz import set_train_mode


@variational_estimator
class MultiLayerPerceptron(TabularModel, Transfer):
    @use_kwargs_dict(
        ps=None,
        embed_p=0.0,
        y_range=None,
        use_bn=True,
        bn_final=False,
        act_cls=ReLU(inplace=True),
        embedding_module=None,
    )
    def __init__(
        self,
        ann_structure,
        emb_sz=[],
        embedding_module=None,
        final_activation=Identity,
        bn_cont=True,
        **kwargs
    ):
        """

        Parameters
        ----------
        ann_structure : list of integers
            amount of features for each layer.
        emb_sz : list
            currently not used.
        embedding_module : dies.embedding
            if not 'None', use the given embedding module and adjust the network accordingly.
        final_activation : class
            activation function for the last layer.
        bn_cont : bool
            decide whether a batch norm is to be used for the continuous data.
        kwargs :

        """
        n_cont = ann_structure[0]
        if embedding_module is not None:
            emb_sz = []
            ann_structure[0] = ann_structure[0] + embedding_module.no_of_embeddings

        self.embedding_module = embedding_module
        self.final_activation = final_activation()

        super(MultiLayerPerceptron, self).__init__(
            emb_sz,
            ann_structure[0],
            ann_structure[-1],
            ann_structure[1:-1],
            bn_cont=bn_cont,
            **kwargs
        )

        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None

    def forward(self, categorical_data, continuous_data):
        """

        Parameters
        ----------
        categorical_data : pytorch.Tensor
            categorical input data. only used when an embedding module is available.
        continuous_data : pytorch.Tensor
            continuous input data.

        Returns
        -------
        pytorch.Tensor
            concatenated outputs of the network for continuous and categorical data.
        """
        if self.embedding_module is None:
            x = super().forward(categorical_data, continuous_data)
        else:
            categorical_data = self.embedding_module(categorical_data)
            if self.bn_cont is not None:
                continuous_data = self.bn_cont(continuous_data)

            x = torch.cat([categorical_data, continuous_data], 1)
            x = self.layers(x)

        return self.final_activation(x)

    def network_split(self):
        "Default split of the body and head"

        if self.embedding_module is not None:
            splitter = lambda m: L(
                m.layers[0],
                m.embedding_module,
                m.layers[1:-1],
                m.layers[-1:],
            ).map(params)

            lr = L(1e-6, 1e-6, 1e-6, 1e-4)
        else:
            splitter = lambda m: L(
                m.layers[0],
                m.layers[1:-1],
                m.layers[-1:],
            ).map(params)

            lr = L(1e-6, 1e-6, 1e-4)

        return splitter, lr

    def train(self, mode: bool = True):
        super().train(mode)
        set_train_mode(self, mode)
