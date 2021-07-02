from dies.utils_blitz import convert_layer_to_bayesian
import random
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as mse

import unittest

import torch
from torch.nn.modules import conv

from dies.data import (
    tp_from_df_from_dtypes,
    tp_from_df,
    get_y_ranges,
)
from fastai.learner import Learner
from fastai.metrics import rmse
from fastai.losses import MSELossFlat

from dies import data
from dies.mlp import MultiLayerPerceptron
from dies.embedding import Embedding, EmbeddingModule, get_emb_sz_list
from dies.utils_pytorch import dev_to_np, xavier_init_uniform
from dies.autoencoder import Autoencoder
from dies.cnn import TemporalCNN, TemporalConvNet
from dies.multitask_architectures import CSNetwork, HPSNetwork
from dies.utils_pytorch import np_to_dev, dev_to_np
from dies.losses import CnnMSELoss
from dies.recurrent_models import LSTMModel
from sklearn import datasets
from dies.utils_blitz import convert_layer_to_bayesian, convert_to_bayesian_model

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


class TestMLP(unittest.TestCase):
    def setUp(self):
        n_features = 3
        device = "cpu"

        df = get_df(50, 2)
        tp = tp_from_df_from_dtypes(df, "target", valid_percent=0.2, standardize_X=True)
        self.dls = tp.dataloaders(bs=16)

        set_random_states()

    def test_simple_mlp(self):
        input_size = self.dls.train_ds.conts.shape[1]

        ann_model = MultiLayerPerceptron(
            ann_structure=[input_size, 5, 1],
        )
        learn = Learner(self.dls, ann_model)

        self.eval(learn)

    def eval(self, learn, n_epochs=1):
        targets = self.dls.train_ds.ys.values

        learn.recorder.silent = True
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        e_init = mse(targets, y_hat)
        learn.fit(n_epochs)
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        e_end = mse(targets, y_hat)

        self.assertLess(e_end, e_init)

    def test_mlp_with_yrange(self):
        input_size = self.dls.train_ds.conts.shape[1]

        y_ranges = get_y_ranges(self.dls.train_ds.ys)

        ann_model = MultiLayerPerceptron(
            ann_structure=[input_size, 10, 2, 1],
            y_range=y_ranges[0],
        )

        learn = Learner(self.dls, ann_model, lr=1e-4, loss_func=torch.nn.MSELoss())

        self.eval(learn, n_epochs=1)

    def test_simple_mlp_with_embedding(self):
        input_size = self.dls.train_ds.conts.shape[1]

        ann_model = MultiLayerPerceptron(
            ann_structure=[input_size, 2, 1],
            emb_sz=get_emb_sz_list([11, 11]),
            embed_p=0.1,
        )

        learn = Learner(self.dls, ann_model, loss_func=torch.nn.MSELoss())

        self.eval(learn)

    def test_bayes_mlp(self):
        input_size = self.dls.train_ds.conts.shape[1]
        ann_model = MultiLayerPerceptron(
            ann_structure=[input_size, 1, 1], embedding_module=None
        )
        ann_model = convert_to_bayesian_model(ann_model)
        learn = Learner(self.dls, ann_model, lr=10, loss_func=torch.nn.MSELoss())

        self.eval(learn)

    def test_true(self):
        self.assertTrue(True)


class TestAE(unittest.TestCase):
    def setUp(self):
        n_features = 3
        device = "cpu"

        self.df = get_df(50, 2)

        set_random_states()

    def eval(self, learn, targets, n_epochs=1):
        learn.recorder.silent = True
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        e_init = mse(targets, y_hat)
        learn.fit(n_epochs)
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        e_end = mse(targets, y_hat)

        self.assertLess(e_end, e_init)

    def test_simple_ae(self):
        set_random_states()
        cols = ["feat1", "feat2"]
        tp = tp_from_df(
            self.df,
            y_columns=cols,
            x_columns=cols,
            standardize_X=True,
            valid_percent=0.2,
        )
        dls = tp.dataloaders(bs=16)
        targets = dls.train_ds.ys.values
        input_size = dls.train_ds.conts.shape[1]

        ann_structure = [20, 4, 1]

        ann_model = Autoencoder(ann_structure=[input_size] + ann_structure)

        learn = Learner(dls, ann_model, loss_func=torch.nn.MSELoss())

        self.eval(learn, targets, n_epochs=1)

    def test_ae_with_yranges(self):
        set_random_states()
        cols = ["feat1", "feat2"]
        tp = tp_from_df(
            self.df,
            y_columns=cols,
            x_columns=cols,
            standardize_X=True,
            valid_percent=0.2,
        )
        dls = tp.dataloaders(bs=16)
        targets = dls.train_ds.ys.values
        input_size = dls.train_ds.conts.shape[1]
        y_ranges = get_y_ranges(dls.train_ds.ys)

        ann_structure = [20, 4, 2]

        ann_model = Autoencoder(
            ann_structure=[input_size] + ann_structure, y_ranges=y_ranges
        )
        ann_model.apply(xavier_init_uniform)

        learn = Learner(dls, ann_model, loss_func=torch.nn.MSELoss())

        self.eval(learn, targets, n_epochs=1)

    def test_ae_with_embedding_and_yrange(self):
        set_random_states()
        cols = ["feat1", "feat2"]
        tp = tp_from_df(
            self.df,
            y_columns=cols,
            x_columns=cols,
            cat_columns=["cat_1", "cat_2"],
            standardize_X=True,
            valid_percent=0.2,
        )
        dls = tp.dataloaders(bs=16)
        targets = dls.train_ds.ys.values
        input_size = dls.train_ds.conts.shape[1]
        y_ranges = get_y_ranges(dls.train_ds.ys)

        ann_structure = [20, 4, 3]
        embedding_module = EmbeddingModule([11, 11], embedding_dropout=0.1)

        ann_model = Autoencoder(
            ann_structure=[input_size] + ann_structure,
            embedding_module=embedding_module,
            y_ranges=y_ranges,
            embeding_position="start",
        )

        learn = Learner(dls, ann_model, lr=0.01, loss_func=torch.nn.MSELoss())

        self.eval(learn, targets, n_epochs=1)

    def test_ae_with_embedding_at_bottleneck(self):
        set_random_states()
        cols = ["feat1", "feat2"]
        tp = tp_from_df(
            self.df,
            y_columns=cols,
            x_columns=cols,
            cat_columns=["cat_1", "cat_2"],
            standardize_X=True,
            valid_percent=0.2,
        )
        dls = tp.dataloaders(bs=16)
        targets = dls.train_ds.ys.values
        input_size = dls.train_ds.conts.shape[1]
        y_ranges = get_y_ranges(dls.train_ds.ys)

        ann_structure = [20, 4, 1]
        embedding_module = EmbeddingModule([11, 11], embedding_dropout=0.1)

        ann_model = Autoencoder(
            ann_structure=[input_size] + ann_structure,
            embedding_module=embedding_module,
            y_ranges=y_ranges,
            embeding_position="bottleneck",
        )

        learn = Learner(dls, ann_model, loss_func=torch.nn.MSELoss())

        self.eval(learn, targets, n_epochs=1)

    def test_true(self):
        self.assertTrue(True)

    def test_yranges_mtl(self):
        # TODO: does y ranges work with mtl e.g. in mtl, mlp, and cnn architectures
        pass


class TestCNN(unittest.TestCase):
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

    def test_simple_cnn(self):
        ann_structure = [5, 10, 10, 1]
        ann_model = TemporalCNN(
            ann_structure,
            kernel_size=3,
            embedding_module=None,
            cnn_type="cnn",
        )
        learn = Learner(self.dls, ann_model, loss_func=CnnMSELoss())
        self.eval(learn)

    def test_simple_cnn_embedding(self):
        ann_structure = [5, 10, 10, 1]
        embedding_module = EmbeddingModule([11, 11], embedding_dropout=0.1)
        ann_model = TemporalCNN(
            ann_structure,
            kernel_size=3,
            embedding_module=embedding_module,
            cnn_type="cnn",
        )
        learn = Learner(self.dls, ann_model, loss_func=CnnMSELoss())
        self.eval(learn)

    def test_temporal_cnn(self):
        ann_structure = [5, 10, 10, 1]
        ann_model = TemporalCNN(ann_structure, kernel_size=3, batch_norm_cont=True)
        learn = Learner(self.dls, ann_model, loss_func=CnnMSELoss())
        self.eval(learn)

    def test_temporal_cnn_embedding(self):
        ann_structure = [5, 10, 10, 1]
        embedding_module = EmbeddingModule([11, 11], embedding_dropout=0.1)
        ann_model = TemporalCNN(
            ann_structure,
            kernel_size=3,
            embedding_module=embedding_module,
            batch_norm_cont=True,
        )
        learn = Learner(self.dls, ann_model, loss_func=CnnMSELoss())
        self.eval(learn)

    def test_bayes_cnn(self):
        ann_structure = [5, 10, 10, 1]
        ann_model = TemporalCNN(
            ann_structure,
            kernel_size=3,
            embedding_module=None,
            batch_norm_cont=True,
        )
        ann_model = convert_to_bayesian_model(ann_model)
        learn = Learner(self.dls, ann_model, loss_func=CnnMSELoss())
        self.eval(learn)


# class TestTemporalConvNet(unittest.TestCase):
#     def setUp(self):
#         n_features = 3
#         device = "cpu"
#         df = get_df(2400, 5)
#         con_cols = [f"feat{i+1}" for i in range(5)]
#         cat_cols = ["cat_1", "cat_2"]
#         self.tp = tp_from_df(
#             df,
#             y_columns=["target"],
#             x_columns=con_cols,
#             cat_columns=cat_cols,
#             standardize_X=True,
#             do_split_by_n_weeks=True,
#         )
#
#         train_tl = data.TimeseriesTransform(
#             self.tp,
#             timeseries_length=24,
#             batch_first=True,
#             sequence_last=True,
#             is_train=True,
#             drop_inconsistent_cats=False,
#         )
#
#         valid_tl = data.TimeseriesTransform(
#             self.tp,
#             timeseries_length=24,
#             batch_first=True,
#             sequence_last=True,
#             is_train=False,
#             is_valid=True,
#             drop_inconsistent_cats=False,
#         )
#         self.dls = data.DataLoaders.from_dsets(
#             train_tl.to_tfmd_lists(), valid_tl.to_tfmd_lists(), bs=10
#         )
#
#         set_random_states()


class TestCS(unittest.TestCase):
    def setUp(self):
        n_features = 40
        device = "cpu"

        df = get_df(50, n_features, 2)
        tp = tp_from_df_from_dtypes(
            df, ["target1", "target2"], valid_percent=0.2, standardize_X=True
        )
        self.dls = tp.dataloaders(bs=8)

    def eval(self, learn, targets, n_epochs=1):
        learn.recorder.silent = True
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        e_init = mse(targets, y_hat)

        learn.fit(n_epochs, lr=1)
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        e_end = mse(targets, y_hat)
        self.assertLess(e_end, e_init)

    def test_simple_CS(self):
        targets = self.dls.train_ds.ys.values

        layer_params = [[30, 5, 2, 1], [10, 5, 2, 1]]
        number_of_tasks = len(layer_params)
        cs_model = CSNetwork(layer_params, number_of_tasks)

        learn = Learner(self.dls, cs_model, loss_func=torch.nn.MSELoss())
        self.eval(learn, targets, 5)

    def test_CS_embedding(self):
        targets = self.dls.train_ds.ys.values

        layer_params = [[30, 5, 2, 1], [10, 5, 2, 1]]
        embedding_module = EmbeddingModule([11, 11], embedding_dropout=0.1)
        number_of_tasks = len(layer_params)
        cs_model = CSNetwork(
            layer_params, number_of_tasks, embedding_module=embedding_module
        )

        learn = Learner(self.dls, cs_model, loss_func=torch.nn.MSELoss())
        self.eval(learn, targets, 5)


class TestHPS(unittest.TestCase):
    def setUp(self):
        n_features = 40
        device = "cpu"

        df = get_df(50, n_features, 2)
        tp = tp_from_df_from_dtypes(
            df, ["target1", "target2"], valid_percent=0.2, standardize_X=True
        )
        self.dls = tp.dataloaders(bs=8)

    def eval(self, learn, targets, n_epochs=1):
        learn.recorder.silent = True
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        e_init = mse(targets, y_hat)

        learn.fit(n_epochs, lr=10)
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        e_end = mse(targets, y_hat)
        self.assertLess(e_end, e_init)

    def test_simple_HPS(self):
        targets = self.dls.train_ds.ys.values

        shared_layer_params = [40, 20, 10]
        separate_layer_params = [10, 5, 2, 1]
        number_of_tasks = 2
        hps_model = HPSNetwork(
            shared_layer_params, separate_layer_params, number_of_tasks
        )

        learn = Learner(self.dls, hps_model, loss_func=torch.nn.MSELoss())
        self.eval(learn, targets, 5)

    def test_HPS_embedding(self):
        targets = self.dls.train_ds.ys.values

        shared_layer_params = [40, 20, 10]
        separate_layer_params = [10, 5, 2, 1]
        number_of_tasks = 2
        embedding_module = EmbeddingModule([11, 11], embedding_dropout=0.1)
        hps_model = HPSNetwork(
            shared_layer_params,
            separate_layer_params,
            number_of_tasks,
            embedding_module=embedding_module,
        )

        learn = Learner(self.dls, hps_model, loss_func=torch.nn.MSELoss())
        self.eval(learn, targets, 5)


class TestLSTM(unittest.TestCase):
    def setUp(self):
        n_features = 40
        device = "cpu"

        df = get_df(1000, n_features, 1)
        tp = tp_from_df_from_dtypes(
            df, ["target"], valid_percent=0.2, standardize_X=True
        )
        tp_train = data.TimeseriesTransform(
            tp,
            timeseries_length=50,
            drop_inconsistent_cats=False,
            is_train=True,
            sequence_last=True,
        )
        tp_valid = data.TimeseriesTransform(
            tp,
            timeseries_length=50,
            drop_inconsistent_cats=False,
            is_valid=True,
            sequence_last=True,
        )

        self.dls = data.DataLoaders.from_dsets(
            tp_train.to_tfmd_lists(), tp_valid.to_tfmd_lists(), bs=10
        )

    def eval(self, learn, targets, n_epochs=1):
        learn.recorder.silent = True
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        # Remove dimension of 'y_hat' to match 'targets'
        y_hat = y_hat[:, 0]
        e_init = mse(targets, y_hat)

        learn.fit(n_epochs, lr=10)
        y_hat, _ = dev_to_np(learn.get_preds(ds_idx=0))
        y_hat = y_hat[:, 0]
        e_end = mse(targets, y_hat)
        self.assertLess(e_end, e_init)

    def test_LSTM(self):
        # Remove dimension of 'targets', as sklearn.mse supports two dimensions only
        targets = self.dls.train_ds.ys[:, 0]

        ann_model = LSTMModel(
            timeseries_length=50,
            n_features_hidden_state=5,
            num_recurrent_layer=3,
            output_dim=50,
        )

        learn = Learner(self.dls, ann_model, loss_func=torch.nn.MSELoss())
        self.eval(learn, targets, 1)