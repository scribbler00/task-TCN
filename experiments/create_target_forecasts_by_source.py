import os
import tqdm

SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = "4"
    os.environ["SLURM_CPUS_PER_TASK"] = SLURM_CPUS_PER_TASK
    SERVER = False
else:
    SERVER = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)
torch.set_num_threads(1)
from multiprocessing import Pool
import argparse

import json
import pandas as pd
from fastcore.foundation import L

from fastai.losses import MSELossFlat
from fastai.learner import Learner
from fastai.metrics import rmse
from fastai.learner import *

from dies.utils import set_random_states

from confer.utils import (
    adapt_dict_path_to_backend,
    add_args,
    get_data_type_and_year,
    get_embedding,
    get_error,
    get_target_files,
    preselect_keys,
    str_to_path,
)
from rep.utils_data import (
    TargetData,
    TargetDataTimeSeries,
)

# default_device = "cpu"

pd.options.mode.chained_assignment = None


def train(target_files, source_model_file, base_folder, run_id, model_type):
    print(target_files)
    with open(
        base_folder / f"park_id_to_name_{run_id}_{model_type}.json"
    ) as input_file:
        task_id_to_name = json.load(input_file)
        adapt_dict_path_to_backend(task_id_to_name, SERVER)

    for target_file in target_files:
        dfs_cur_target = L()
        source_learn = load_learner(source_model_file)
        for cur_task_id in task_id_to_name.keys():
            cur_task_id = int(cur_task_id)
            set_random_states(42)
            if "mlp" in source_model_file.stem.lower():
                target_data = TargetData(
                    target_file,
                    batch_size=512,
                    cols_to_drop=["Month", "Hour"],
                    task_id=cur_task_id,
                    set_minimal_validation=False,
                    valid_percent=0,
                    num_days_training=365,
                    shuffle_train=False,
                    old_drop_cols=True,
                )
            else:
                target_data = TargetDataTimeSeries(
                    target_file,
                    batch_size=512,
                    cols_to_drop=["Month", "Hour"],
                    task_id=cur_task_id,
                    set_minimal_validation=False,
                    valid_percent=0,
                    num_days_training=365,
                    shuffle_train=False,
                    old_drop_cols=True,
                )
            if target_data.n_train_samples == 0:
                continue

            learn = Learner(
                target_data.train_dataloaders[0],
                source_learn.model,
                loss_func=MSELossFlat(),
            )

            # learn.cbs = [c for c in learn.cbs if type(c) == TrainEvalCallback]

            _, preds_train = get_error(learn, 0, return_forecasts=True)
            _, preds_test = get_error(
                learn, 0, test_dl=target_data.test_dataloader[0], return_forecasts=True
            )

            df_train = pd.DataFrame(target_data.df_train.PowerGeneration)
            df_test = pd.DataFrame(target_data.df_test.PowerGeneration)
            df_train["Yhat"] = preds_train
            df_test["Yhat"] = preds_test
            df_train["TaskID"] = cur_task_id
            df_test["TaskID"] = cur_task_id

            df_comb = pd.concat([df_train, df_test], axis=0)
            dfs_cur_target += df_comb

        if len(dfs_cur_target) > 0:
            dfs_cur_target = pd.concat(dfs_cur_target, axis=0)
            dfs_cur_target.to_csv(
                base_folder
                / f"preds_source/{target_file.stem}_{source_model_file.stem}.csv",
                sep=";",
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    add_args(parser)

    parser.add_argument(
        "--model_type",
        help="tcn, cnn, or mlp.",
        default="tcn",
    )

    parser.add_argument(
        "--device",
        help="cpu or cuda:0",
        default="cpu",
    )

    parser.add_argument(
        "--emb_at_ith_layer", help="first/last/first_and_last,none,all", default="first"
    )

    args = parser.parse_args()
    data_folder = str_to_path(args.data_folder)
    full_bayes = True if args.full_bayes.lower() == "yes" else False

    embedding_type = args.embedding_type
    model_type = args.model_type.lower()
    device = args.device
    run_id = args.run_id
    emb_at_ith_layer = args.emb_at_ith_layer.lower()

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    embedding_type = get_embedding(args.embedding_type.lower())
    year, data_type = get_data_type_and_year(data_folder, args.gefcom_type)

    result_folder = str_to_path(args.result_folder + f"/{year}_{data_type}")
    str_to_path(str(result_folder) + "/preds_source", create_folder=True)

    with open(f"{result_folder}/splits.json") as f:
        splits = json.load(f)

    all_files = data_folder.ls()

    adapt_dict_path_to_backend(splits, SERVER)

    for cur_run_id in tqdm.tqdm(preselect_keys(splits, run_id)):
        print(f"Start cur_run_id {cur_run_id}.")
        cur_source_files = splits[cur_run_id]

        target_files = get_target_files(all_files, cur_source_files, data_type)

        source_model_file = (
            result_folder
            / f"{model_type}_source_model_{full_bayes}_{cur_run_id}_{embedding_type}_residuals_{emb_at_ith_layer}.pkl"
        ).absolute()

        def train_wrapper(file):
            train(
                # target_files,
                [file],
                source_model_file,
                result_folder,
                cur_run_id,
                model_type,
            )

        with Pool(processes=min(SLURM_CPUS_PER_TASK, len(target_files)) - 1) as pool:
            pool.map(train_wrapper, target_files)

        print(f"End cur_run_id {cur_run_id}.")
