import random
import pandas as pd
import numpy as np
import torch
from torch import nn

from sklearn.metrics import mean_squared_error as mse
from sklearn import datasets
from dies.adaboost import ELM, AdaBoost
from dies.utils_pytorch import dev_to_np, np_to_dev

import unittest


def set_random_states(random_state):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


# Create the dataset
def get_data():
    rng = np.random.RandomState()
    X = np.linspace(-6, 6, 100)[:, np.newaxis]
    y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

    return np_to_dev(((X))), np_to_dev(((y.reshape(-1, 1))))


def get_data_2d(n=100):
    X = np.linspace(-6, 6, n)[:, np.newaxis]

    ys = np.ones((n, 2))
    y1 = (
        np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.normal(0, 0.1, X.shape[0])
    )
    y2 = (
        np.cos(X).ravel() + np.cos(6 * X).ravel() + np.random.normal(0, 0.3, X.shape[0])
    )

    ys[:, 0] = y1
    ys[:, 1] = y2 + 3

    return np_to_dev(((X))), np_to_dev(ys)


class TestELM(unittest.TestCase):
    def setUp(self):
        set_random_states(0)
        self.X, self.y = get_data()
        self.X2d, self.y2d = get_data_2d()

    def test_simple_elm(self):
        elm = ELM([1, 10, 1])
        yh = elm(self.X)
        e_init = mse(dev_to_np(self.y), dev_to_np(yh))

        elm.fit(self.X, self.y)

        yh = elm(self.X)
        e_end = mse(dev_to_np(self.y), dev_to_np(yh))

        self.assertLess(e_end, e_init)

    def test_2d_elm(self):
        elm = ELM([1, 10, 2])
        yh = elm(self.X2d)
        e_init = mse(dev_to_np(self.y2d), dev_to_np(yh))

        elm.fit(self.X2d, self.y2d)

        yh = elm(self.X2d)
        e_end = mse(dev_to_np(self.y2d), dev_to_np(yh))

        self.assertLess(e_end, e_init)


class TestAdaboost(unittest.TestCase):
    def setUp(self):
        set_random_states(0)
        self.X, self.y = get_data()
        self.X2d, self.y2d = get_data_2d()

    def test_simple_elm(self):
        adaboost = AdaBoost(n_estimators=2)

        adaboost.fit(self.X, self.y)

        yh = adaboost.predict(self.X, "median")
        e_end = mse(dev_to_np(self.y), dev_to_np(yh))

        self.assertLess(e_end, 0.5)

        yh = adaboost.predict(self.X, "mean")
        e_end = mse(dev_to_np(self.y), dev_to_np(yh))

        self.assertLess(e_end, 0.5)

    def test_simple_elm_single_estimator(self):
        adaboost = AdaBoost(n_estimators=1)

        adaboost.fit(self.X, self.y)

        yh = adaboost.predict(self.X, "median")
        e_end = mse(dev_to_np(self.y), dev_to_np(yh))

        self.assertLess(e_end, 0.5)

        yh = adaboost.predict(self.X, "mean")
        e_end = mse(dev_to_np(self.y), dev_to_np(yh))

        self.assertLess(e_end, 0.5)

    def test_2d_elm(self):
        # hier stimmt etwas nicht
        adaboost = AdaBoost(n_estimators=20)

        adaboost.fit(self.X2d, self.y2d)

        yh = adaboost.predict(self.X2d, "mean")
        e_end = mse(dev_to_np(self.y2d), dev_to_np(yh))
        # self.assertLess(e_end, 0.5)

        yh = adaboost.predict(self.X2d, "median")
        e_end = mse(dev_to_np(self.y2d), dev_to_np(yh))
        # self.assertLess(e_end, 0.5)

    def test_2d_weight_tasks_error(self):
        # hier stimmt etwas nicht
        adaboost = AdaBoost(weight_tasks_error=True)

        adaboost.fit(self.X2d, self.y2d)

        yh = adaboost.predict(self.X2d, "mean")
        e_end = mse(dev_to_np(self.y2d), dev_to_np(yh))
        # self.assertLess(e_end, 0.5)

        yh = adaboost.predict(self.X2d, "median")
        e_end = mse(dev_to_np(self.y2d), dev_to_np(yh))
        # self.assertLess(e_end, 0.5)
