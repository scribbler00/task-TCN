__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


import numpy as np

import torch
from torch import nn
from dies.data import np_to_dev

from sklearn.base import BaseEstimator


class ELM(torch.nn.Module):
    def __init__(self, ann_structure):
        super(ELM, self).__init__()
        if len(ann_structure) != 3:
            raise ValueError("ELM can only have [input_size, n_hidden, output_size]")

        input_size = ann_structure[0]
        n_hidden = ann_structure[1]
        output_size = ann_structure[-1]

        self.linear1 = torch.nn.Linear(input_size, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden, output_size)

        torch.nn.init.xavier_uniform_(
            self.linear1.weight, gain=nn.init.calculate_gain("relu")
        )

    def fit(self, X, y):
        with torch.no_grad():
            X_1 = self.forward_l1(X)
            Xt = X_1.T
            W_out = (Xt @ X_1).pinverse() @ (Xt @ y)
            self.linear2.weight = torch.nn.Parameter(W_out.T)

            return self

    def forward_l1(self, x):
        with torch.no_grad():
            return torch.relu_(self.linear1(x))

    def predict(self, x):
        return self.forward(x)

    def forward(self, x):
        with torch.no_grad():
            out = self.forward_l1(x)
            out = self.linear2(out)
            return out


class AdaBoost(BaseEstimator):
    def __init__(
        self,
        n_estimators=10,
        n_hidden=100,
        loss="linear",
        lr=1e-3,
        weight_tasks_error=False,
    ):
        super(AdaBoost, self).__init__()

        self.n_estimators = n_estimators
        self.n_hidden = n_hidden
        self.lr = lr
        self.loss = loss
        self.weight_tasks_error = weight_tasks_error

    def boost(self, iboost, X, y, sample_weight):
        # based on "Boosting for Regression Transfer" by David Pardoe and Peter Stone in ICML10
        # Algorithm 1

        estimator = ELM([X.shape[1], self.n_hidden, y.shape[1]])

        n_samples = X.shape[0]
        # Step 1:
        bootstrap_idx = np_to_dev(
            np.random.choice(
                np.arange(n_samples), size=n_samples, replace=True, p=sample_weight
            )
        )

        X_ = X[bootstrap_idx]
        y_ = y[bootstrap_idx]

        estimator.fit(X_, y_)

        #  Step 2: Calculate the adjusted error e_{i}^{t} for each instance:

        #  calculates \abs{y_j-h_t(x_j)}
        if self.n_tasks == 1:
            error_vect = torch.abs(estimator(X) - y).reshape(-1)
        else:
            # extension for multi-task problems
            error_vect = torch.abs(estimator(X) - y)

            if self.weight_tasks_error:
                error_vect = error_vect * (
                    error_vect.sum(dim=0) / error_vect.sum(dim=0).sum()
                )
            # normalize by the number of tasks
            error_vect = error_vect.sum(dim=1) / self.n_tasks

        # mask for current samples
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        # D_t = max^{n}_{j=1}|y_j-h_t(x_j)
        error_max = masked_error_vector.max()

        if error_max != 0:
            # e_{i}^{t} = \abs{y_i - h_t(x_j)} / D_t
            masked_error_vector /= error_max

        # extension for different losses. TODO: should this before the normalization?
        if self.loss == "square":
            masked_error_vector = masked_error_vector ** 2
        elif self.loss == "exponential":
            masked_error_vector = 1.0 - torch.exp(-masked_error_vector)

        # Step 3: Calculate the adjusted error of h_t:

        # \epsilon_t = \sum_{i=1}^{n}{e_i^{t} w_i^{t}}
        estimator_error = (masked_sample_weight * masked_error_vector).sum()

        if estimator_error <= 0:
            # stop if fit is perfect
            return sample_weight, 1.0, 0.0
        elif estimator_error >= 0.5:
            # TODO remove estimator if it is not the only one
            # raise NotImplementedError
            print("This should bre retrained, but not yet implemented.")

        #  Step 4: \beta_t = \epsilon_t/(1-\epsilon_t)
        beta = estimator_error / (1.0 - estimator_error)

        # Step 5: Update the weight vector:
        # TODO: This is actually not in the paper. Do we need it?
        estimator_weight = self.lr * torch.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= torch.pow(
                beta, (1.0 - masked_error_vector) * self.lr
            )

        return sample_weight, estimator_weight, estimator_error, estimator

    def fit(self, X, y, sample_weight=None):

        n_samples = X.shape[0]
        self.n_tasks = y.shape[1]

        if sample_weight is None:
            sample_weight = torch.ones(n_samples, dtype=torch.float64)

        sample_weight /= sample_weight.sum()

        self.estimator_weights = torch.zeros(self.n_estimators, dtype=torch.float64)
        self.estimator_errors = torch.ones(self.n_estimators, dtype=torch.float64)

        self.estimators = []
        with torch.no_grad():
            for iboost in range(self.n_estimators):
                # Step 1 - 5:
                (
                    sample_weight,
                    estimator_weight,
                    estimator_error,
                    estimator,
                ) = self.boost(iboost, X, y, sample_weight)

                self.estimators.append(estimator)

                # early termination
                if sample_weight is None:
                    break

                self.estimator_weights[iboost] = estimator_weight
                self.estimator_errors[iboost] = estimator_error

                if estimator_error == 0:
                    break

                # Step 5: Normalize by Z_t
                sample_weight_sum = sample_weight.sum()

                if iboost < self.n_estimators - 1:
                    # normalize
                    sample_weight /= sample_weight_sum

            return self

    def predict(self, X, forecast_type="median"):
        with torch.no_grad():
            if forecast_type == "mean":
                yhats = np.ones((len(X), self.n_tasks))
                for t_id in range(self.n_tasks):
                    yh = (
                        torch.cat(
                            [est(X)[:, t_id].reshape(-1, 1) for est in self.estimators],
                            axis=1,
                        )
                        * (self.estimator_weights / self.estimator_weights.sum())
                    ).sum(dim=1)

                    yhats[:, t_id] = yh

                return yhats

            elif forecast_type == "median":
                yhats = np.ones((len(X), self.n_tasks))
                for t_id in range(self.n_tasks):
                    predictions = torch.cat(
                        [est(X)[:, t_id].reshape(-1, 1) for est in self.estimators],
                        axis=-1,
                    )
                    sorted_idx = torch.argsort(predictions, axis=1)
                    weight_cdf = torch.cumsum(
                        self.estimator_weights[sorted_idx], axis=1
                    )
                    median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1].reshape(
                        -1, 1
                    )

                    median_idx = median_or_above.type(torch.uint8).argmin(dim=-1)
                    # argmin takes the last smallest value, so this is the one for the median
                    if self.n_estimators > 2:
                        # in case there is only a single estimator we can't access the next value/median
                        median_idx += 1

                    median_idx[median_idx >= self.n_estimators] = self.n_estimators

                    median_estimators = sorted_idx[np.arange(len(X)), median_idx]

                    yh = predictions[np.arange(len(X)), median_estimators]

                    yhats[:, t_id] = yh

                return yhats
