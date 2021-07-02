import random
import numpy as np
from sklearn import datasets
import pandas as pd
from dies.mlp import MultiLayerPerceptron
import unittest
from fastai.torch_core import to_device, tensor
from dies.losses import *
from dies.embedding import EmbeddingModule, EmbeddingType
from dies.autoencoder import VariationalAutoencoder
import torch

random_state = 0


def set_random_states():
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def get_df():
    X, y, _ = datasets.make_regression(
        n_samples=50,
        n_features=2,
        bias=1000,
        n_informative=2,
        noise=10,
        coef=True,
        random_state=42,
    )

    df1 = pd.DataFrame(
        data=np.concatenate([X, y.reshape(-1, 1)], axis=1),
        columns=["feat1", "feat2", "target"],
    )
    cats = np.random.randint(low=0, high=10, size=(df1.shape[0], 2))
    df1["cat_1"] = cats[:, 0]
    df1["cat_2"] = cats[:, 1]

    index1 = pd.date_range("2000-01-01", "2000-06-01", periods=df1.shape[0])
    index1 = pd.to_datetime(index1, utc=True)
    df1.index = index1

    return df1


class TestVILoss(unittest.TestCase):
    def setUp(self):
        n_features = 3
        device = "cpu"

        df = get_df()

        set_random_states()
        self.X = tensor(df[["feat1", "feat2"]].values)
        self.cat = tensor(df[["cat_1", "cat_2"]].values)
        self.y = tensor(df[["target"]].values)

        self.embedding_module = EmbeddingModule(
            [],
            embedding_dropout=0,
            embedding_type=EmbeddingType.Bayes,
            embedding_dimensions=[(10, 2), (10, 2)],
        )

    def test_KLDLoss(self):
        input_size = 2
        lambd = 0.1
        ann_model = MultiLayerPerceptron(
            ann_structure=[input_size, 5, 1], embedding_module=self.embedding_module
        )

        y_hat = ann_model(self.cat, self.X)
        loss_func = VILoss(ann_model, kl_weight=lambd)

        loss = loss_func(y_hat, self.y)
        loss.backward()

        embeds = ann_model.embedding_module.embeddings
        for emb in embeds:
            self.assertIsNotNone(emb.weight_sampler.mu.grad)
            self.assertIsNotNone(emb.weight_sampler.rho.grad)

    def test_vi_loss(self):
        input_size = 2

        ann_model = MultiLayerPerceptron(
            ann_structure=[input_size, 5, 1], embedding_module=self.embedding_module
        )

        y_hat = ann_model(self.cat, self.X)

        loss_func = VILoss(ann_model)

        loss = loss_func(y_hat, self.y)
        loss.backward()

        embeds = ann_model.embedding_module.embeddings

    def test_vi_loss_bnn(self):
        input_size = 2

        ann_model = MultiLayerPerceptron(
            ann_structure=[input_size, 5, 1], embedding_module=self.embedding_module
        )

        y_hat = ann_model(self.cat, self.X)

        loss_func = VILoss(ann_model)

        loss = loss_func(y_hat, self.y)
        loss.backward()

        embeds = ann_model.embedding_module.embeddings
        for emb in embeds:
            self.assertIsNotNone(emb.weight_sampler.mu.grad)
            self.assertIsNotNone(emb.weight_sampler.rho.grad)

    def testCnnMSELoss(self):
        y = torch.ones([5, 5, 5])
        y_hat = torch.ones([5, 5, 5]) / 2

        loss_func = CnnMSELoss()
        loss = loss_func(y_hat, y)
        self.assertTrue(loss == 0.25)

    def testVAEReconstructionLoss(self):
        ann_model = VariationalAutoencoder(
            [2, 1], embedding_module=self.embedding_module
        )
        X_hat = ann_model(self.cat, self.X)
        loss_func = VAEReconstructionLoss(ann_model)
        loss = loss_func(self.X, X_hat)
        loss.backward()

        embeds = ann_model.embedding_module.embeddings
        for emb in embeds:
            self.assertIsNotNone(emb.weight_sampler.mu.grad)
            self.assertIsNotNone(emb.weight_sampler.rho.grad)

    def testRSSLoss(self):
        y = torch.ones([5, 5, 5])
        y_hat = torch.ones([5, 5, 5]) / 2

        loss_func = RSSLoss()
        loss = loss_func(y_hat, y)
        self.assertTrue(loss == 31.25)
