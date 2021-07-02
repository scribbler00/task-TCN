import copy
import torch
from fastai.callback.core import Callback
from fastai.layers import Embedding
from fastai.losses import MSELossFlat
from fastcore.basics import listify

import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as sk_mse
from fastcore.foundation import L

from fastai.learner import Learner
from fastai.metrics import rmse
from fastai.learner import *
from confer.utils import (
    get_error,
    filter_source_preds,
)

from fastai.callback.all import *

pd.options.mode.chained_assignment = None
from blitz.modules.embedding_bayesian_layer import BayesianEmbedding


def copy_cat_embedding(embedding_mod, emb_id, from_cat_ids: list, to_cat_ids: list):
    from_cat_ids, to_cat_ids = listify(from_cat_ids), listify(to_cat_ids)
    with torch.no_grad():
        emb = embedding_mod.embeddings[emb_id]
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


def increase_embedding_by_one(embedding_mod, emb_id: int, device="cpu"):
    with torch.no_grad():
        emb = embedding_mod.embeddings[emb_id]

        if isinstance(emb, Embedding):
            emb_new = Embedding(emb.num_embeddings + 1, emb.embedding_dim).to(device)
            elements_to_copy = list(range(emb.weight.shape[0]))
            emb_new.weight[elements_to_copy, :] = emb.weight[elements_to_copy, :]
        elif isinstance(emb, BayesianEmbedding):
            emb_new = BayesianEmbedding(emb.num_embeddings + 1, emb.embedding_dim).to(
                device
            )
            elements_to_copy = list(range(emb.weight_sampler.mu.shape[0]))
            emb_new.weight_sampler.mu[elements_to_copy, :] = emb.weight_sampler.mu[
                elements_to_copy, :
            ]
            emb_new.weight_sampler.rho[elements_to_copy, :] = emb.weight_sampler.rho[
                elements_to_copy, :
            ]
        else:
            raise ValueError("Unexpected embedding type.")

        embedding_mod.embeddings[emb_id] = emb_new


def store_preds(
    base_folder,
    config,
    cur_season,
    dls_test,
    emb_at_ith_layer,
    emb_type,
    full_bayes,
    is_mlp_arch,
    model_type,
    num_days_training,
    preds,
    target_file,
):
    if is_mlp_arch:
        df_preds = dls_test.train_ds.items[["PowerGeneration"]]
    else:
        df_preds = dls_test.tp.items[["PowerGeneration"]]
    df_preds["Preds"] = preds

    # TODO naming
    df_preds.to_csv(
        base_folder
        / f"preds/{model_type}_{target_file.stem}_fullBayes_{full_bayes}_embType_{emb_type}_numDays_{num_days_training}_residual_{emb_at_ith_layer}_embAdaption_{config['embedding_adaption']}_ftType_{config['ft_type']}_curSeason_{cur_season}.csv",
        sep=";",
    )


def get_configs(config):
    configs = []
    for epochs in [1, 2, 5, 10, 20]:
        for wd in [0, 0.25, 0.5]:
            new_conf = copy.copy(config)
            new_conf["epochs"] = epochs
            new_conf["wd"] = wd
            configs.append(new_conf)
    return configs


def get_errors(learn, dls_test, device, return_forecasts=False):
    learn.model.eval()
    val_error = get_error(learn, 1, return_forecasts=False, device=device)
    res = get_error(
        learn, 1, test_dl=dls_test[0], device=device, return_forecasts=return_forecasts
    )
    if return_forecasts:
        return val_error, res[0], res[1]
    else:
        return val_error, res


def adapt_model(best_task_id, config, device, learn):
    if (
        config["embedding_adaption"] == "reset"
        and learn.model.embedding_module is not None
    ):
        increase_embedding_by_one(learn.model.embedding_module, emb_id=0, device=device)
    elif (
        config["embedding_adaption"] == "copy"
        and learn.model.embedding_module is not None
    ):
        increase_embedding_by_one(learn.model.embedding_module, emb_id=0, device=device)
        copy_cat_embedding(learn.model.embedding_module, 0, best_task_id, -1)
    learn.model.to(device)


def get_learner(device, dls_trains, source_model_file):
    source_learn = load_learner(source_model_file)
    learn = Learner(
        dls_trains[0],
        source_learn.model.to(device),
        loss_func=MSELossFlat(),
    )
    return learn


def get_data_loader(
    best_task_id,
    config,
    device,
    is_mlp_arch,
    months_to_train_on,
    num_days_training,
    standardize,
    target_data_type,
    target_file,
    valid_percent,
):
    dls_test, dls_trains = None, None

    target_data = target_data_type(
        target_file,
        num_days_training=num_days_training,
        batch_size=config["batch_size"],
        cols_to_drop=["Month", "Hour"],
        task_id=best_task_id,
        set_minimal_validation=False,
        valid_percent=valid_percent,
        standardize=standardize,
        months_to_train_on=months_to_train_on,
        n_folds=1,
        old_drop_cols=True,
    )
    if target_data.n_train_samples == 0:
        target_data = None
    elif (
        (target_data.n_train_samples < num_days_training * 0.8) and not is_mlp_arch
    ) or (
        (
            (target_data.n_train_samples // target_data.n_samples_per_day)
            < num_days_training * 0.8
        )
        and is_mlp_arch
    ):
        target_data = None

    if target_data is not None:
        target_data.batch_size = max(target_data.n_train_samples // 10, 1)

        dls_trains = target_data.train_dataloaders
        dls_test = target_data.test_dataloader

        if (
            len(dls_trains[0].train_ds) == 0
            or len(dls_trains[0].valid_ds) == 0
            or len(dls_test.train_ds) == 0
        ):
            dls_trains, dls_test = None, None

        for idx in range(len(dls_trains)):
            dls_trains[idx] = dls_trains[idx].to(device)
        dls_test = dls_test.to(device)

    return dls_test, dls_trains


def get_best_task_id(
    best_error,
    best_task_id,
    df_source_orig,
    months_to_train_on,
    num_days_training,
    park_id_to_name,
    split_date,
    valid_percent,
):
    for cur_task_id in park_id_to_name.keys():
        cur_task_id = int(cur_task_id)
        df_source = filter_source_preds(
            copy.copy(df_source_orig),
            cur_task_id,
            num_days_training,
            split_date,
            months_to_train_on,
        )

        if len(df_source) == 0:
            break

        df_source_sampled = df_source.sample(
            frac=valid_percent, replace=False, random_state=12
        )
        val_error_before_ft = (
            sk_mse(
                df_source_sampled.PowerGeneration,
                df_source_sampled.Yhat,
            )
            ** 0.5
        )

        if val_error_before_ft < best_error:
            best_error = val_error_before_ft
            best_task_id = cur_task_id
    return best_task_id


def get_lambd_from_config(
    embedding_type, model_type, emb_at_ith_layer, result_folder, cur_run_id
):
    config_file = (
        str(result_folder)
        + f"/{model_type}_{embedding_type}_{emb_at_ith_layer}_{cur_run_id}_config.csv"
    )
    df_config = pd.read_csv(config_file)
    df_config = df_config.sort_values("root_mean_squared_error").iloc[0]
    lambd = df_config["config/lambd"]

    return lambd