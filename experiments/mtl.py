import os

import tqdm


SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = "4"
    os.environ["SLURM_CPUS_PER_TASK"] = SLURM_CPUS_PER_TASK
    SERVER = False
else:
    SERVER = True

os.environ["OMP_NUM_THREADS"] = SLURM_CPUS_PER_TASK  # "1"
os.environ["MKL_NUM_THREADS"] = SLURM_CPUS_PER_TASK  # "1"
os.environ["NUMEXPR_NUM_THREADS"] = SLURM_CPUS_PER_TASK  # "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)
torch.set_num_threads(SLURM_CPUS_PER_TASK)

import argparse
import json
import pandas as pd

from fastcore.foundation import L
from fastai.learner import Learner
from fastai.metrics import rmse
from fastai.learner import *
from fastai.losses import MSELossFlat
from fastai.callback.all import *

# from fastai.torch_core import default_device

from dies.embedding import EmbeddingType

from confer.utils_training import get_cnn_config, get_mlp_config
from confer.utils import (
    adapt_dict_path_to_backend,
    add_args,
    get_data_type_and_year,
    get_error,
    is_mlp,
    preselect_keys,
    str_to_path,
)
from rep.utils_data import (
    TargetData,
    TargetDataTimeSeries,
)


# default_device = "cpu"

pd.options.mode.chained_assignment = None

from confer.utils_similarity import *


def create_results(
    source_files,
    source_model_file,
    base_folder,
    run_id,
    model_type,
    full_bayes,
    emb_at_ith_layer,
    year,
):
    standardize = True
    source_learn = load_learner(source_model_file)

    target_data_type = (
        TargetDataTimeSeries if not is_mlp(source_model_file) else TargetData
    )

    # emb_type = "bayes" if "bayes" in str(source_model_file).lower() else "normal"

    with open(
        base_folder / f"park_id_to_name_{run_id}_{model_type}.json"
    ) as input_file:
        park_id_to_name = json.load(input_file)
        adapt_dict_path_to_backend(park_id_to_name, SERVER)
        task_name_to_id = {v[0]: int(k) for k, v in park_id_to_name.items()}

    mtl_errors, files = L(), L()
    df_res = pd.DataFrame({})
    for cur_source_file in source_files:
        if cur_source_file not in list(task_name_to_id.keys()):
            continue

        cur_task_id = task_name_to_id[cur_source_file]

        cur_target_data = target_data_type(
            cur_source_file,
            num_days_training=365,
            batch_size=config["batch_size"],
            cols_to_drop=["Month", "Hour", "LeadTime"],
            task_id=cur_task_id,
            set_minimal_validation=False,
            valid_percent=0,
            standardize=standardize,
            # months_to_train_on=months_to_train_on,
            n_folds=1,
            old_drop_cols=True,
            minimum_training_fraction=0.1,
        )

        learn = Learner(
            cur_target_data.train_dataloaders[0],
            source_learn.model.to(device),
            loss_func=MSELossFlat(),
        )

        zero_shot_error = get_error(
            learn,
            1,
            test_dl=cur_target_data.test_dataloader[0],
            return_forecasts=False,
        )

        mtl_errors += zero_shot_error
        files += cur_source_file
        df_res = pd.DataFrame(
            {
                "ZeroShotError": mtl_errors,
                "Files": files,
            }
        )

    return df_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    add_args(parser)

    parser.add_argument(
        "--model_type",
        help="tcn, cnn, mlp",
        default="tcn",
    )

    parser.add_argument(
        "--device",
        help="cpu or cuda:0.",
        default="cpu",
    )

    parser.add_argument(
        "--emb_at_ith_layer",
        help="first/last/first_and_last,none,all",
        default="all",
    )

    args = parser.parse_args()
    device = args.device
    run_id = args.run_id
    full_bayes = False
    for embedding_type in [EmbeddingType.Normal, EmbeddingType.Bayes]:
        for dataset in [
            "PVSandbox2015",
            "WindSandbox2015",
        ]:
            data_folder = "/".join(args.data_folder.split("/")[0:-2]) + "/" + dataset
            data_folder = str_to_path(data_folder)
            for emb_at_ith_layer in ["first", "all"]:
                for model_type in ["tcn", "mlp"]:

                    if full_bayes and embedding_type == EmbeddingType.Bayes:
                        continue

                    if model_type == "mlp" and emb_at_ith_layer != "first":
                        continue

                    if (
                        emb_at_ith_layer == "none"
                        and embedding_type == EmbeddingType.Bayes
                    ):
                        continue
                    print(
                        dataset,
                        model_type,
                        embedding_type,
                        emb_at_ith_layer,
                    )
                    year, data_type = get_data_type_and_year(
                        data_folder, args.gefcom_type
                    )

                    result_folder = str_to_path(
                        args.result_folder + f"/{year}_{data_type}"
                    )
                    str_to_path(str(result_folder) + "/preds", create_folder=True)

                    with open(f"{result_folder}/splits.json") as f:
                        splits = json.load(f)

                    all_files = data_folder.ls()

                    if is_mlp(model_type):
                        config = get_mlp_config(year, data_type, embedding_type)
                    else:
                        config = get_cnn_config(
                            year, data_type, embedding_type, emb_at_ith_layer
                        )

                    adapt_dict_path_to_backend(splits, SERVER)

                    str_to_path(str(result_folder) + "/mtl", create_folder=True)

                    dfs_mtl_res = []
                    for cur_run_id in tqdm.tqdm(preselect_keys(splits, run_id)):

                        cur_source_files = splits[cur_run_id]

                        source_model_file = (
                            result_folder
                            / f"{model_type}_source_model_{full_bayes}_{cur_run_id}_{embedding_type}_residuals_{emb_at_ith_layer}.pkl"
                        ).absolute()

                        df_mtl_res = create_results(
                            cur_source_files,
                            source_model_file,
                            result_folder,
                            cur_run_id,
                            model_type,
                            full_bayes,
                            emb_at_ith_layer,
                            year,
                        )
                        dfs_mtl_res.append(df_mtl_res)

                    dfs_mtl_res = pd.concat(dfs_mtl_res, axis=0).reset_index()

                    print(dfs_mtl_res.describe())

                    dfs_mtl_res.to_csv(
                        result_folder
                        / f"mtl/{model_type}__fullBayes_{full_bayes}_embType_{embedding_type}_residual_{emb_at_ith_layer}_mtl_results.csv",
                        sep=";",
                    )
