__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


import pandas as pd
import numpy as np
from enum import Enum


import torch
import torch
import torch.nn as nn

from fastai.tabular.model import emb_sz_rule
from fastai.basics import *
from fastai.layers import *


from blitz.modules.embedding_bayesian_layer import BayesianEmbedding
from blitz.utils import variational_estimator


class EmbeddingType(Enum):
    Normal = 0
    Bayes = 1


def get_emb_sz_list(dims: list):
    """
    For all elements in the given list, find a size for the respective embedding through trial and error
    Each element denotes the amount of unique values for one categorical feature
    Parameters
    ----------
    dims : list
        a list containing a number of integers.
    Returns
    -------
    list of tupels
        a list containing an the amount of unique values and respective embedding size for all elements.
    """
    return [(d, emb_sz_rule(d)) for d in dims]


@variational_estimator
class EmbeddingModule(nn.Module):
    def __init__(
        self,
        categorical_dimensions,
        embedding_dropout=0.0,
        embedding_dimensions=None,
        embedding_type=EmbeddingType.Normal,
    ):
        super().__init__()
        """
        Parameters
        ----------
        categorical_dimensions : list of integers
            List with number of categorical values for each feature. 
            Output size is calculated based on 
            fastais `emb_sz_rule`. In case explicit dimensions
            are required use `embedding_dimensions`.

        embedding_dropout : Float
            the dropout to be used after the embedding layers.

        embedding_dimensions : list of tupels
            This list will contain a two element tuple for each
            categorical feature. The first element of a tuple will
            denote the number of unique values of the categorical
            feature. The second element will denote the embedding
            dimension to be used for that feature. If None, `categorical_dimensions`
            is used to determine the dimensions.
        embedding_type : integer
            the type of embedding that is to be used.
            0 : Normal
            1 : Bayes
            2 : BayesBNN
        """

        self.embedding_type = embedding_type

        # Set the embedding dimension for all features
        if embedding_dimensions is None:
            self.embedding_dimensions = get_emb_sz_list(categorical_dimensions)
        else:
            self.embedding_dimensions = embedding_dimensions

        self.embeddings = nn.ModuleList(
            [self._embedding(ni, nf) for ni, nf in self.embedding_dimensions]
        )
        self.emb_drop = nn.Dropout(embedding_dropout) if embedding_dropout > 0 else None

    @property
    def _embedding(self):
        if self.embedding_type == EmbeddingType.Normal:
            emb_type = Embedding
        elif self.embedding_type == EmbeddingType.Bayes:

            def bayes_bnn(ni, nf):
                return BayesianEmbedding(ni, nf)

            emb_type = bayes_bnn
        else:
            raise ValueError(f"Unknown embedding type {self.embedding_type}.")

        return emb_type

    @property
    def no_of_embeddings(self):
        return sum(e.embedding_dim for e in self.embeddings)

    @property
    def categorical_dimensions(self):
        return L(e.num_embeddings for e in self.embeddings)

    def forward(self, categorical_data):
        """

        Parameters
        ----------
        categorical_data : pytorch.Tensor
            categorical input data.

        Returns
        -------
        pytorch.Tensor
            concatenated outputs of the network for all categorical features.
        """
        x = torch.cat(
            [
                emb_layer(categorical_data[:, i])
                for i, emb_layer in enumerate(self.embeddings)
            ],
            1,
        )

        x = self.emb_drop(x) if self.emb_drop is not None else x

        return x

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters of the embeddings.
        Parameters
        ----------
        device (:class:`torch.device`) : the desired device of the parameters
                and buffers in this module
        dtype (:class:`torch.dtype`) : the desired floating point type of
            the floating point parameters and buffers in this module
        tensor (torch.Tensor) : Tensor whose dtype and device are the desired
            dtype and device for all parameters and buffers in this module
        memory_format (:class:`torch.memory_format`) : the desired memory
            format for 4D parameters and buffers in this module (keyword
            only argument)

        Returns
        -------
        self
        """
        self = super().to(*args, **kwargs)
        for idx, emb in enumerate(self.embeddings):
            self.embeddings[idx] = emb.to(*args, **kwargs)

        return self

    def reset_cat_embedding(self, emb_id: int, cat_ids: list):
        cat_ids = listify(cat_ids)
        with torch.no_grad():
            emb = self.embeddings[emb_id]
            reset_data = self._embedding(emb.num_embeddings, emb.embedding_dim)
            if self.embedding_type == EmbeddingType.Normal:
                emb.weight[cat_ids, :] = reset_data.weight[cat_ids, :]
            else:
                emb.weight_sampler.mu[cat_ids, :] = reset_data.weight_sampler.mu[
                    cat_ids, :
                ]
                emb.weight_sampler.rho[cat_ids, :] = reset_data.weight_sampler.rho[
                    cat_ids, :
                ]

    def copy_cat_embedding(self, emb_id, from_cat_ids: list, to_cat_ids: list):
        from_cat_ids, to_cat_ids = listify(from_cat_ids), listify(to_cat_ids)
        with torch.no_grad():
            emb = self.embeddings[emb_id]
            for idx in range(len(from_cat_ids)):
                if isinstance(emb, Embedding):
                    emb.weight[to_cat_ids[idx], :] = emb.weight[from_cat_ids[idx], :]
                elif isinstance(emb, BayesianEmbedding):
                    emb.weight_sampler.mu[to_cat_ids[idx], :] = emb.weight_sampler.mu[
                        from_cat_ids[idx], :
                    ]
                    emb.weight_sampler.rho[to_cat_ids[idx], :] = emb.weight_sampler.rho[
                        from_cat_ids[idx], :
                    ]
                else:
                    raise ValueError("Unexpected embedding type.")

    def increase_embedding_by_one(self, emb_id: int):
        with torch.no_grad():
            emb = self.embeddings[emb_id]
            emb_new = self._embedding(emb.num_embeddings + 1, emb.embedding_dim)

            if isinstance(emb, Embedding):
                elements_to_copy = list(range(emb.weight.shape[0]))
                emb_new.weight[elements_to_copy, :] = emb.weight[elements_to_copy, :]
            elif isinstance(emb, BayesianEmbedding):
                elements_to_copy = list(range(emb.weight_sampler.mu.shape[0]))
                emb_new.weight_sampler.mu[elements_to_copy, :] = emb.weight_sampler.mu[
                    elements_to_copy, :
                ]
                emb_new.weight_sampler.rho[
                    elements_to_copy, :
                ] = emb.weight_sampler.rho[elements_to_copy, :]
            else:
                raise ValueError("Unexpected embedding type.")

            self.embeddings[emb_id] = emb_new
