from enum import Enum
from logging import debug
import os
from fastai.callback.training import BnFreeze
from fastcore.foundation import L
from fastcore.utils import parallel
from matplotlib.pyplot import grid
import numpy as np
import ray
from sklearn.metrics import mean_squared_error as sk_mse
import time
import torch
from ray import tune

from fastai.learner import Learner, load_learner
from fastai.metrics import MSELossFlat
from fastai.metrics import rmse
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.torch_core import params, to_np
from fastai.callback.core import TrainEvalCallback

from dies.utils_pytorch import unfreeze, freeze
from dies.losses import CnnMSELoss, RSSLoss, VILoss
from dies.cnn import *
from dies.utils_pytorch import unfreeze as dies_unfreeze
from dies.embedding import EmbeddingType
from dies.utils_pytorch import print_requires_grad
from dies.gan import *
from dies.mlp import MultiLayerPerceptron
from fastai.layers import Embedding
from fastai.callback.core import Callback
from confer.utils import get_error


def create_and_fit_learner(
    dls,
    model,
    config,
    embedding_type,
    model_type="mlp",
    silent=False,
):
    if model_type == "mlp":
        base_loss = MSELossFlat()
    else:
        base_loss = MSELossFlat()

    if embedding_type == EmbeddingType.Bayes:
        loss = VILoss(
            model,
            base_loss=base_loss,
            kl_weight=config["lambd"],
        )
    else:
        loss = base_loss

    cbs = []
    learn = Learner(dls, model, loss_func=loss, metrics=[rmse], cbs=cbs)

    if silent:
        learn.cbs = L(c for c in learn.cbs if type(c) == TrainEvalCallback)
    else:
        print(learn.model)

    if config["use_one_cycle"]:
        learn.fit_one_cycle(25, wd=config["wd"])

    learn.fit(config["epochs"], lr=config["lr"], wd=config["wd"])

    return learn


def fine_tune(
    source_model_file,
    dls_train,
    lr,
    wd,
    epochs,
    lambd=None,
    silent=False,
    ft_type="emb_only",  # or emb_first_last, disc_lr
    learn=None,
    device="cpu",
):

    if learn is None:
        learn = load_learner(source_model_file)

    freeze(learn.model)

    if isinstance(learn.model, TemporalCNN):
        splitter, disc_lr = learn.model.network_split()

        disc_lr = lr = L(1e-5, 1e-4, 1e-6, 1e-5)

        base_loss = RSSLoss()
        if ft_type == "emb_first_last":
            unfreeze(learn.model.layers.temporal_blocks[0])
            unfreeze(learn.model.layers.temporal_blocks[-1])
        if learn.model.embedding_module is not None:
            unfreeze(learn.model.embedding_module)
        else:
            disc_lr = lr = L(1e-5, 1e-6, 1e-5)

    elif isinstance(learn.model, MultiLayerPerceptron):
        freeze(learn.model)
        splitter, disc_lr = learn.model.network_split()
        disc_lr = lr = L(1e-5, 1e-4, 1e-6, 1e-5)

        if ft_type == "emb_first_last":
            # unfreeze first and last layer
            unfreeze(learn.model.layers[0])
            unfreeze(learn.model.layers[-1])
        if learn.model.embedding_module is not None:
            unfreeze(learn.model.embedding_module)
        else:
            disc_lr = lr = L(1e-5, 1e-6, 1e-5)

        base_loss = MSELossFlat()
    else:
        raise NotImplementedError

    if "bayes" in str(source_model_file).lower():
        loss = VILoss(
            learn.model,
            base_loss=base_loss,
            kl_weight=lambd,
        )
    else:
        loss = base_loss

    learn = Learner(
        dls_train, learn.model, loss_func=loss, metrics=[rmse], splitter=splitter
    )

    if silent:
        learn.cbs = L(c for c in learn.cbs if type(c) == TrainEvalCallback)

    # train
    if epochs > 0:
        if ft_type == "disc_lr":
            unfreeze(learn.model)
            learn.fit(epochs, wd=wd, lr=disc_lr)
        else:
            learn.fit(epochs, wd=wd, lr=lr)

    return learn


def ray_log_nrmse(nrmse):
    tune.report(root_mean_squared_error=nrmse)


def ray_log(learn, full_bayes=False, n_samples=100):
    preds, targets = learn.get_preds(ds_idx=1)
    if full_bayes:
        preds = [preds]
        for i in range(0, n_samples - 1):

            preds_i, _ = learn.get_preds(ds_idx=1)
            preds.append(preds_i)
        preds = torch.cat(preds, 1).mean(1)
    preds, targets = (
        to_np(preds.reshape(-1)),
        to_np(targets.reshape(-1)),
    )

    if np.isnan(preds.sum()):
        cur_error = 1e18
    else:
        cur_error = sk_mse(preds, targets, squared=False)

    tune.report(root_mean_squared_error=cur_error)


def save_learner(
    learn,
    result_folder,
    embedding_type,
    run_id,
    model_name,
    full_bayes,
    emb_at_ith_layer,
):
    learn.path = result_folder
    et = str(embedding_type)
    learn.export(
        f"{model_name}_source_model_{full_bayes}_{run_id}_{et}_residuals_{emb_at_ith_layer}.pkl"
    )


def get_mlp_config(year, data_type, embedding_type):
    config = {
        "lr": 1e-4,
        "use_one_cycle": True,
        "wd": 0.0,
        "epochs": 100,
        "batch_size": 1024,
        "size_multiplier": 20,
        "percental_reduce": 50,
        "dropout": 0.1,
        "embedding_dropout": 0.0,
        "keep_month": False,
        "lambd": 1e-6,
        "gradient_clipping": None,
        "emb_at_ith_layer": "first",
    }

    if embedding_type == EmbeddingType.Bayes:
        config["epochs"] = 200

    return config


def get_mlp_config_ray(
    year,
    data_type,
    embedding_type,
):
    config = get_mlp_config(
        year,
        data_type,
        embedding_type,
    )
    config["lambd"] = tune.grid_search([1e-1, 1e-3, 1e-6])
    config["epochs"] = tune.grid_search([50, 100])
    config["dropout"] = tune.grid_search([0, 0.25, 0.5])
    config["size_multiplier"] = tune.grid_search([1, 5, 10, 20])
    config["batch_size"] = tune.grid_search([1024, 2048, 4096])

    return config


def get_cnn_config_ray(year, data_type, embedding_type, emb_at_ith_layer):
    config = get_cnn_config(year, data_type, embedding_type, emb_at_ith_layer)
    config["lambd"] = tune.grid_search([1e-1, 1e-3, 1e-6])
    config["epochs"] = tune.grid_search([50, 100])
    config["dropout"] = tune.grid_search([0, 0.25, 0.5])
    config["size_multiplier"] = tune.grid_search([1, 5, 10, 20])
    config["batch_size"] = tune.grid_search([32, 64, 128])

    return config


def get_cnn_config(year, data_type, embedding_type, emb_at_ith_layer):
    config = {
        "lr": 1e-4,  # non bayes
        "use_one_cycle": True,
        "wd": 0.0,
        "epochs": 100,
        "dropout": 0.25,
        "embedding_dropout": 0.0,
        "lambd": 0.01,
        "keep_month": False,
        "gradient_clipping": None,
        "percental_reduce": 50,
    }

    # receptive_field (l) = 2^l-1, where l is the number of layers and a kernel size of 3 is used [YA Farha 2019]
    if year == "2015" and data_type == "pv":
        config["num_layers"] = 4
        config["size_multiplier"] = 2
        config["batch_size"] = 256
        config["epochs"] = 25
    elif year == "2015" and data_type == "wind":
        config["num_layers"] = 5
        config["size_multiplier"] = 15
        config["batch_size"] = 32
        config["dropout"] = 0.1
    elif year == "2017" and data_type == "wind":
        config["num_layers"] = 5
        config["size_multiplier"] = 10
        config["batch_size"] = 256
        config["dropout"] = 0.1
    elif year == "2017" and data_type == "pv":
        config["num_layers"] = 3
        config["size_multiplier"] = 5
        config["batch_size"] = 1024
        config["epochs"] = 25
    elif year == "2014" and data_type == "pv":
        config["num_layers"] = 5
        config["size_multiplier"] = 5
        config["batch_size"] = 32
        config["epochs"] = 25
    elif year == "2014" and data_type == "wind":
        config["num_layers"] = 5
        config["size_multiplier"] = 5
        config["batch_size"] = 512

    if embedding_type == EmbeddingType.Bayes and emb_at_ith_layer.lower() != "none":
        config["epochs"] = 200

    config["emb_at_ith_layer"] = emb_at_ith_layer

    return config
