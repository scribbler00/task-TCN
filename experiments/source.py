import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = "4"
    SERVER = False
else:
    SERVER = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)
torch.set_num_threads(SLURM_CPUS_PER_TASK)


# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# torch.set_num_threads(1)


import argparse
import pandas as pd
import numpy as np
import json
from ray import tune
import ray
import torch
from fastai.metrics import rmse
import json
from dies.utils import set_random_states
from dies.embedding import EmbeddingType


from confer.utils_models import get_cnn_model, get_mlp_model
from confer.utils import (
    adapt_dict_path_to_backend,
    add_args,
    get_data_type_and_year,
    get_embedding,
    get_error,
    is_full_bayes,
    is_mlp,
    str_to_path,
)
from confer.utils_training import (
    get_cnn_config_ray,
    get_mlp_config_ray,
    save_learner,
    create_and_fit_learner,
    ray_log_nrmse,
)
from dies.utils_blitz import convert_to_bayesian_model
from rep.utils_data import SourceData, SourceDataTimeSeries

pd.options.mode.chained_assignment = None


def train(
    files,
    config,
    result_folder,
    embedding_type,
    run_id,
    model_type="cnn",
    silent=False,
    full_bayes=False,
    n_folds=1,
    standardize=True,
    save_model=True,
):
    set_random_states(42)

    source_data_type = SourceDataTimeSeries if not is_mlp(model_type) else SourceData

    source_data = source_data_type(
        files,
        n_folds=n_folds,
        batch_size=config["batch_size"],
        cols_to_drop=["Month", "Hour"],
        valid_percent=0.1,
        standardize=standardize,
        old_drop_cols=True,
    )
    n_input_features = source_data.n_input_features
    dls = source_data.train_dataloaders

    files = source_data.files

    learn = None
    dl = dls[0]
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    dl = dl.to(device)

    if is_mlp(model_type):
        park_id_to_name = {
            str(dls[0].items.TaskID.unique()[i]): str(source_data.files[i])
            for i in range(len(source_data.files))
        }

        model = get_mlp_model(
            n_input_features,
            config,
            source_data.categorical_dimensions,
            embedding_type,
        )
    else:
        park_id_to_name = {
            str(dls[0].train_ds.tp.items.TaskID.unique()[i]): str(source_data.files[i])
            for i in range(len(source_data.files))
        }
        model = get_cnn_model(
            n_input_features,
            config,
            source_data.categorical_dimensions,
            embedding_type,
            cnn_type=model_type,
        )

    if full_bayes:
        convert_to_bayesian_model(model.layers)

    learn = create_and_fit_learner(
        dl,
        model,
        config,
        embedding_type,
        model_type=model_type,
        silent=silent,
    )

    nrmse = get_error(learn, 1, device=device)

    print(f"Validation nRMSE: {nrmse}")
    if save_model:
        save_learner(
            learn,
            result_folder,
            embedding_type,
            run_id,
            model_name=model_type,
            full_bayes=full_bayes,
            emb_at_ith_layer=config["emb_at_ith_layer"],
        )
    else:
        ray_log_nrmse(nrmse)

    with open(
        result_folder / f"park_id_to_name_{run_id}_{model_type}.json", "w"
    ) as outfile:
        json.dump(park_id_to_name, outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    add_args(parser)

    parser.add_argument(
        "--model_type",
        help="tcn, cnn, or mlp.",
        default="tcn",
    )

    parser.add_argument(
        "--emb_at_ith_layer", help="first/last/first_and_last,none,all", default="first"
    )

    args = parser.parse_args()
    data_folder = str_to_path(args.data_folder)
    full_bayes = is_full_bayes(args.full_bayes)
    emb_at_ith_layer = args.emb_at_ith_layer.lower()
    model_type = args.model_type.lower()

    embedding_type = get_embedding(args.embedding_type.lower())

    year, data_type = get_data_type_and_year(data_folder, args.gefcom_type)

    result_folder = str_to_path(args.result_folder + f"/{year}_{data_type}")

    if model_type == "mlp":
        config_ray = get_mlp_config_ray(year, data_type, embedding_type)
    else:
        config_ray = get_cnn_config_ray(
            year, data_type, embedding_type, emb_at_ith_layer
        )
    if embedding_type == EmbeddingType.Normal:
        config_ray.pop("lambd", None)

    with open(f"{result_folder}/splits.json") as f:
        splits = json.load(f)

    adapt_dict_path_to_backend(splits, SERVER)
    for run_id in list(splits.keys()):

        files = splits[run_id]

        def train_wrapper(config):
            train(
                files,
                config,
                result_folder,
                embedding_type,
                run_id,
                model_type=model_type,
                full_bayes=full_bayes,
                silent=True,
                save_model=False,
            )

        temp_folder = "/mnt/work/transfer/tmp/"
        if not os.path.exists(temp_folder):
            temp_folder = "/tmp/"

        set_random_states(42)
        ray.init(num_cpus=SLURM_CPUS_PER_TASK)

        if torch.cuda.is_available():
            resources_per_trial = {"gpu": 0.2}
        else:
            resources_per_trial = {"cpu": 4}

        num_samples = 1
        analysis = tune.run(
            train_wrapper,
            name=f"{model_type}_{emb_at_ith_layer}_{run_id}",
            config=config_ray,
            verbose=1,
            resources_per_trial=resources_per_trial,
            num_samples=num_samples,
            max_failures=2,
            raise_on_failed_trial=False,
            metric="root_mean_squared_error",
            mode="min",
        )

        analysis.dataframe().to_csv(
            f"{result_folder}/{model_type}_{embedding_type}_{emb_at_ith_layer}_{run_id}_config.csv"
        )

        ray.shutdown()

        train(
            files,
            analysis.get_best_config(metric="root_mean_squared_error", mode="min"),
            result_folder,
            embedding_type,
            run_id,
            model_type=model_type,
            full_bayes=full_bayes,
            silent=False,
        )
