import copy
from multiprocessing import Pool
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""


SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = "4"
    os.environ["SLURM_CPUS_PER_TASK"] = SLURM_CPUS_PER_TASK
    SERVER = False
else:
    SERVER = True

os.environ["OMP_NUM_THREADS"] = SLURM_CPUS_PER_TASK
os.environ["MKL_NUM_THREADS"] = SLURM_CPUS_PER_TASK
os.environ["NUMEXPR_NUM_THREADS"] = SLURM_CPUS_PER_TASK
SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)

import torch

torch.set_num_threads(SLURM_CPUS_PER_TASK)
import argparse

import json
import pandas as pd
import numpy as np
from fastcore.foundation import L

# from fastai.torch_core import default_device

from dies.utils import set_random_states
from dies.embedding import EmbeddingType

from confer.utils_training import fine_tune, get_cnn_config, get_mlp_config
from confer.utils import (
    adapt_dict_path_to_backend,
    add_args,
    get_data_type_and_year,
    get_embedding,
    get_source_preds,
    get_split_year,
    get_target_files,
    is_full_bayes,
    is_mlp,
    preselect_keys,
    str_to_path,
)
from rep.utils_data import (
    TargetData,
    TargetDataTimeSeries,
)

from fastai.callback.all import *


import confer.utils_target as target_util

pd.options.mode.chained_assignment = None


def train(
    target_files,
    source_files,
    config,
    source_model_file,
    base_folder,
    run_id,
    model_type="cnn",
    n_folds=1,
    device="cpu",
    full_bayes=False,
    emb_at_ith_layer="",
    year=2015,
):
    default_device = device
    standardize = True
    df_res = pd.DataFrame()
    source_learn = target_util.load_learner(source_model_file)
    is_mlp_arch = is_mlp(source_model_file)
    target_data_type = (
        TargetDataTimeSeries if not is_mlp(source_model_file) else TargetData
    )

    # set_random_states(23)
    if source_learn.model.embedding_module is not None:
        max_task_id = source_learn.model.embedding_module.categorical_dimensions[0]
    else:
        max_task_id = len(source_files)

    (
        test_errors_after_ft,
        days_training,
        files,
        source_task_ids,
        test_errors_before_ft,
        val_errors_before_ft,
        val_errors_after_ft,
        season,
        epochs_s,
        wds,
    ) = (
        L(),
        L(),
        L(),
        L(),
        L(),
        L(),
        L(),
        L(),
        L(),
        L(),
    )

    with open(
        base_folder / f"park_id_to_name_{run_id}_{model_type}.json"
    ) as input_file:
        park_id_to_name = json.load(input_file)
        adapt_dict_path_to_backend(park_id_to_name, SERVER)

    emb_type = "bayes" if "bayes" in str(source_model_file).lower() else "normal"

    t_year = get_split_year(year)  # int(year)  # - 1
    split_date = pd.to_datetime(f"{t_year}-01-01", utc=True)
    for target_file in target_files:
        df_source = None
        if emb_at_ith_layer != "none" and config["embedding_adaption"] != "reset":
            df_source_orig = get_source_preds(
                base_folder,
                target_file,
                source_model_file,
            )
            if df_source_orig is None:
                print(f"Missing source preds {target_file}.")
                continue
        else:
            df_source_orig = None

        for cur_season, months_to_train_on in enumerate(
            [
                [
                    12,
                    1,
                    2,
                ],
                [
                    3,
                    4,
                    5,
                ],
                [6, 7, 8],
                [9, 10, 11],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            ]
        ):
            for num_days_training in [7, 14, 30, 60, 90, 365]:
                if (cur_season == 4 and num_days_training != 365) or (
                    num_days_training == 365 and cur_season != 4
                ):
                    continue

                best_error = 1e12
                best_task_id = max_task_id
                val_error_before_ft = -1
                valid_percent = 0.1
                if num_days_training == 7:
                    valid_percent = 0.2

                if (
                    emb_at_ith_layer != "none"
                    and config["embedding_adaption"] != "reset"
                ):
                    best_task_id = target_util.get_best_task_id(
                        best_error,
                        best_task_id,
                        df_source_orig,
                        months_to_train_on,
                        num_days_training,
                        park_id_to_name,
                        split_date,
                        valid_percent,
                    )

                    if best_task_id == max_task_id:
                        break

                set_random_states(12)
                dls_test, dls_trains = target_util.get_data_loader(
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
                )
                if dls_trains is None:
                    continue
                # return dls_trains, dl
                print("best_task_id", best_task_id)
                best_val_error = 1e12

                set_random_states(12)

                learn = target_util.get_learner(device, dls_trains, source_model_file)

                target_util.adapt_model(best_task_id, config, device, learn)

                val_error_before_ft, test_error_before_ft = target_util.get_errors(
                    learn,
                    dls_test,
                    device,
                )

                configs = target_util.get_configs(config)

                for config in configs:
                    learn = fine_tune(
                        source_model_file=source_model_file,
                        dls_train=dls_trains[0],
                        lr=1e-4,
                        wd=config["wd"],
                        epochs=config["epochs"],
                        lambd=config["lambd"],
                        ft_type=config["ft_type"],
                        learn=learn,
                        device=device,
                        silent=True,
                    )

                    (
                        val_error_after_ft,
                        test_error_after_ft,
                        preds,
                    ) = target_util.get_errors(
                        learn, dls_test, device, return_forecasts=True
                    )

                    val_errors_before_ft += val_error_before_ft
                    val_errors_after_ft += val_error_after_ft
                    test_errors_after_ft += test_error_after_ft
                    days_training += num_days_training
                    files += target_file
                    source_task_ids += best_task_id
                    test_errors_before_ft += test_error_before_ft
                    season += cur_season
                    epochs_s += config["epochs"]
                    wds += config["wd"]

                    df_res = pd.DataFrame(
                        {
                            "ValidationErrorsBeforeFT": val_errors_before_ft,
                            "ValidationErrorsAfterFT": val_errors_after_ft,
                            "TestErrorsBeforeFT": test_errors_before_ft,
                            "TestErrorsAfterFT": test_errors_after_ft,
                            "DaysTraining": days_training,
                            "Files": files,
                            "SourceTaskID": source_task_ids,
                            "Season": season,
                            "Epochs": epochs_s,
                            "WeightDecay": wds,
                        }
                    )

                    print(
                        df_res[
                            [
                                "TestErrorsAfterFT",
                                "ValidationErrorsAfterFT",
                                "TestErrorsBeforeFT",
                                "DaysTraining",
                                "Epochs",
                            ]
                        ]
                    )

                    # create only prediction of best hyperparameter
                    if val_error_after_ft < best_val_error:
                        best_val_error = val_error_after_ft

                        target_util.store_preds(
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
                        )

    return df_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    add_args(parser)

    parser.add_argument(
        "--model_type",
        help="tcn, cnn, mlp",
        default="mlp",
    )

    parser.add_argument(
        "--device",
        help="cpu or cuda:0.",
        default="cpu",
    )

    parser.add_argument(
        "--embedding_adaption",
        help="reset/copy",
        default="reset",
    )

    parser.add_argument(
        "--ft_type",
        help="emb_only/emb_first_last/disc_lr",
        default="emb_only",
    )

    parser.add_argument(
        "--emb_at_ith_layer",
        help="first/last/first_and_last,none,all",
        default="first",
    )

    args = parser.parse_args()
    data_folder = str_to_path(args.data_folder)
    full_bayes = is_full_bayes(args.full_bayes)

    embedding_type = args.embedding_type
    model_type = args.model_type.lower()
    device = args.device
    run_id = args.run_id
    emb_at_ith_layer = args.emb_at_ith_layer.lower()

    if model_type == "mlp":
        emb_at_ith_layer = "first"

    embedding_type = get_embedding(args.embedding_type.lower())
    year, data_type = get_data_type_and_year(data_folder, args.gefcom_type)

    result_folder = str_to_path(args.result_folder + f"/{year}_{data_type}")
    str_to_path(str(result_folder) + "/preds", create_folder=True)
    str_to_path(str(result_folder) + "/tl/", create_folder=True)

    with open(f"{result_folder}/splits.json") as f:
        splits = json.load(f)

    all_files = data_folder.ls()

    if is_mlp(model_type):
        config = get_mlp_config(year, data_type, embedding_type)
    else:
        config = get_cnn_config(year, data_type, embedding_type, emb_at_ith_layer)

    config["ft_type"] = args.ft_type.lower()
    config["embedding_adaption"] = args.embedding_adaption.lower()

    adapt_dict_path_to_backend(splits, SERVER)

    df_results = []
    for cur_run_id in preselect_keys(splits, run_id):
        print(f"Start cur_run_id {cur_run_id}")
        cur_source_files = splits[cur_run_id]

        target_files = get_target_files(all_files, cur_source_files, data_type)
        if embedding_type == EmbeddingType.Bayes:
            config["lambd"] = target_util.get_lambd_from_config(
                embedding_type, model_type, emb_at_ith_layer, result_folder, cur_run_id
            )

        source_model_file = (
            result_folder
            / f"{model_type}_source_model_{full_bayes}_{cur_run_id}_{embedding_type}_residuals_{emb_at_ith_layer}.pkl"
        ).absolute()

        for cur_file in target_files:
            df_cur_file = train(
                [cur_file],
                cur_source_files,
                config,
                source_model_file,
                result_folder,
                cur_run_id,
                model_type=model_type,
                device=device,
                full_bayes=full_bayes,
                emb_at_ith_layer=emb_at_ith_layer,
                year=year,
            )
            df_cur_file["RunID"] = cur_run_id
            df_results.append(df_cur_file)

            df_results_concatenated = pd.concat(df_results, axis=0).reset_index()

            df_results_concatenated.to_csv(
                result_folder
                / f"tl/{model_type}_fullBayes_{full_bayes}_embType_{embedding_type}_residual_{emb_at_ith_layer}_embAdaption_{config['embedding_adaption']}_ftType_{config['ft_type']}_results.csv",
                sep=";",
            )

        print(f"End cur_run_id {cur_run_id}")
