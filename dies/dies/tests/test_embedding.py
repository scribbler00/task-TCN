from dies.cnn import TemporalCNN
import random
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as mse

import unittest

import torch

from dies.data import (
    tp_from_df,
)
from fastai.learner import Learner
from fastai.metrics import rmse

from dies import data
from dies.embedding import (
    EmbeddingModule,
    EmbeddingType,
)
from dies.utils_pytorch import dev_to_np
from dies.losses import CnnMSELoss
from sklearn import datasets

random_state = 0


def set_random_states():
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def get_df(n_samples, n_features, n_targets=1):
    X, y, _ = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        bias=1000,
        n_informative=2,
        n_targets=n_targets,
        noise=10,
        coef=True,
        random_state=42,
    )

    columns = [f"feat{i+1}" for i in range(n_features)]
    columns += (
        ["target"] if n_targets == 1 else list(f"target{i+1}" for i in range(n_targets))
    )
    data = None
    if n_targets == 1:
        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    else:
        data = np.concatenate([X, y], axis=1)
    df = pd.DataFrame(
        data=data,
        columns=columns,
    )
    cats = np.random.randint(low=0, high=10, size=(df.shape[0], 2))
    df["cat_1"] = cats[:, 0]
    df["cat_2"] = cats[:, 1]

    index1 = pd.date_range("2000-01-01", "2000-06-01", periods=df.shape[0])
    index1 = pd.to_datetime(index1, utc=True)
    df.index = index1

    return df


class TesEmbedding(unittest.TestCase):
    def setUp(self):
        n_features = 3
        device = "cpu"
        df = get_df(2400, 5)
        con_cols = [f"feat{i+1}" for i in range(5)]
        cat_cols = ["cat_1", "cat_2"]
        self.tp = tp_from_df(
            df,
            y_columns=["target"],
            x_columns=con_cols,
            cat_columns=cat_cols,
            standardize_X=True,
            do_split_by_n_weeks=True,
        )

        train_tl = data.TimeseriesTransform(
            self.tp,
            timeseries_length=24,
            batch_first=True,
            sequence_last=True,
            is_train=True,
            drop_inconsistent_cats=False,
        )

        valid_tl = data.TimeseriesTransform(
            self.tp,
            timeseries_length=24,
            batch_first=True,
            sequence_last=True,
            is_train=False,
            is_valid=True,
            drop_inconsistent_cats=False,
        )
        self.dls = data.DataLoaders.from_dsets(
            train_tl.to_tfmd_lists(), valid_tl.to_tfmd_lists(), bs=10
        )

        set_random_states()

    def eval(self, learn, n_epoch=1):
        targets = self.dls.train_ds.ys

        learn.recorder.silent = True
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        targets = targets.reshape(targets.shape[0], targets.shape[2])
        y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[2])

        e_init = mse(targets, y_hat)

        learn.fit(n_epoch=1)
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[2])
        e_end = mse(targets, y_hat)

        self.assertLess(e_end, e_init)

    def test_simple_embedding(self):
        ann_structure = [5, 10, 10, 1]
        embedding_module = EmbeddingModule(
            [11, 11], embedding_dropout=0.1, embedding_type=EmbeddingType.Normal
        )
        ann_model = TemporalCNN(
            ann_structure, kernel_size=3, embedding_module=embedding_module
        )
        learn = Learner(self.dls, ann_model, loss_func=CnnMSELoss())
        self.eval(learn)

    def test_bayes_embedding(self):
        ann_structure = [5, 10, 10, 1]
        embedding_module = EmbeddingModule(
            [11, 11], embedding_dropout=0.1, embedding_type=EmbeddingType.Bayes
        )
        ann_model = TemporalCNN(
            ann_structure, kernel_size=3, embedding_module=embedding_module
        )
        learn = Learner(self.dls, ann_model, loss_func=CnnMSELoss())
        self.eval(learn)

    def test_bayes_bnn_embedding(self):
        ann_structure = [5, 10, 10, 1]
        embedding_module = EmbeddingModule(
            [11, 11], embedding_dropout=0.1, embedding_type=EmbeddingType.Bayes
        )
        ann_model = TemporalCNN(
            ann_structure, kernel_size=3, embedding_module=embedding_module
        )
        learn = Learner(self.dls, ann_model, loss_func=CnnMSELoss())
        self.eval(learn)