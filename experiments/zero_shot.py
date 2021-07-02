import os

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
import numpy as np
import tqdm
import pickle as pkl

from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw


from fastcore.foundation import L
from fastai.losses import MSELossFlat
from fastai.learner import Learner
from fastai.learner import *

from dies.embedding import EmbeddingType


from rep.utils_data import (
    SourceDataTimeSeries,
    TargetData,
    TargetDataTimeSeries,
)

from fastai.callback.all import *

# default_device = "cpu"

pd.options.mode.chained_assignment = None


from rep.utils_preprocessing import most_relevant_feature

from confer.utils_similarity import *
from confer.utils_training import get_cnn_config, get_mlp_config
from confer.utils import (
    adapt_dict_path_to_backend,
    add_args,
    get_data_type_and_year,
    get_error,
    get_target_files,
    is_mlp,
    match_time_stamps,
    preselect_keys,
    str_to_path,
)


def _dtw(x, y):
    distance, path = fastdtw(x.ravel(), y.ravel(), dist=euclidean)
    return distance


def calc_sim(df_1, df_2, sim_measure):
    col = most_relevant_feature(df_1)
    df_1, df_2 = match_time_stamps(df_1, df_2)

    if len(df_1) > 100:
        if sim_measure == "mse":
            res = measure_similarity(df_1, df_2, mean_squared_error, col)[0]
        elif sim_measure == "dtw":
            res = measure_similarity(df_1, df_2, _dtw, col)[0] / len(df_1)
        else:
            res = -1

        res = np.mean(res)
    else:
        print("To short file.")
        res = 1e12

    return res


def get_sim_matrix(all_files, sim_measure):
    if sim_measure == "mp":  # max power
        relevant_files = L()
        max_powers = L()
        for f in all_files:
            with pd.HDFStore(f) as store:
                if "/powerdata" in list(store.keys()):
                    max_power = store["/powerdata"].MaxPowerGeneration.max()
                    relevant_files += f
                    max_powers += max_power
        n_files = len(relevant_files)
        similarities = np.zeros((n_files, n_files))
        max_powers = np.array(max_powers)
        for i in tqdm.tqdm(range(n_files)):
            for j in range(i, n_files):
                if i == j:
                    continue
                similarities[i, j] = (
                    mean_squared_error([max_powers[i]], [max_powers[j]]) ** 0.5
                )
        all_files = relevant_files
    else:
        data_files = L()

        relevant_files = L()
        # read all files
        for file_i in tqdm.tqdm(all_files):
            cur_target_data = SourceDataTimeSeries(
                file_i,
                batch_size=config["batch_size"],
                cols_to_drop=["Month", "Hour", "LeadTime"],
                set_minimal_validation=False,
                valid_percent=None,
                standardize=True,
                n_folds=1,
            )
            if cur_target_data.df_train is not None:
                data_files += cur_target_data
                relevant_files += file_i

        all_files = relevant_files
        n_files = len(all_files)

        similarities = np.zeros((n_files, n_files))
        for i in tqdm.tqdm(range(n_files)):
            for j in range(i, n_files):
                if i == j:
                    continue
                similarities[i, j] = calc_sim(
                    data_files[i].df_train, data_files[j].df_train, sim_measure
                )

    similarities = similarities + similarities.transpose()
    most_similar_parks = np.argpartition(similarities, 1)[:, 1:]
    most_similar_parks_dict = {}
    for idx, cur_file in enumerate(all_files):
        correspondig_files = []
        for file_id in most_similar_parks[idx]:
            correspondig_files.append(all_files[file_id])
        most_similar_parks_dict[cur_file] = correspondig_files

    most_similar_parks = {
        all_files[k]: all_files[v] for k, v in enumerate(most_similar_parks)
    }

    return most_similar_parks


def create_results(
    target_files,
    source_model_file,
    base_folder,
    run_id,
    model_type,
    full_bayes,
    emb_at_ith_layer,
    year,
    most_similar_parks,
    sim_measure,
):
    standardize = True
    source_learn = load_learner(source_model_file)

    target_data_type = (
        TargetDataTimeSeries if not is_mlp(source_model_file) else TargetData
    )

    emb_type = "bayes" if "bayes" in str(source_model_file).lower() else "normal"

    with open(
        base_folder / f"park_id_to_name_{run_id}_{model_type}.json"
    ) as input_file:
        park_id_to_name = json.load(input_file)
        adapt_dict_path_to_backend(park_id_to_name, SERVER)
        park_id_to_name = {v[0]: k for k, v in park_id_to_name.items()}

    zero_shot_errors, files, source_task_ids = L(), L(), L()
    df_res = pd.DataFrame({})
    for cur_target_file in target_files:
        cur_most_similar_task_id = 0
        cur_most_similar_park = most_similar_parks[cur_target_file]
        for a in cur_most_similar_park:
            if a in list(park_id_to_name.keys()):
                cur_most_similar_task_id = int(park_id_to_name[a])
                break

        cur_target_data = target_data_type(
            cur_target_file,
            num_days_training=365,
            batch_size=config["batch_size"],
            cols_to_drop=["Month", "Hour", "LeadTime"],
            task_id=cur_most_similar_task_id,
            set_minimal_validation=False,
            valid_percent=0,
            standardize=standardize,
            # months_to_train_on=months_to_train_on,
            n_folds=1,
            old_drop_cols=True,
            minimum_training_fraction=0.1,
        )

        if cur_target_data.n_train_samples == 0:
            continue

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

        zero_shot_errors += zero_shot_error
        files += cur_target_file
        source_task_ids += cur_most_similar_task_id
        df_res = pd.DataFrame(
            {
                "ZeroShotError": zero_shot_errors,
                "Files": files,
                "SourceTaskID": source_task_ids,
            }
        )

    return df_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    add_args(parser)

    parser.add_argument(
        "--device",
        help="cpu or cuda:0.",
        default="cpu",
    )

    parser.add_argument(
        "--sim_measure",
        help="mse/dtw",
        default="dtw",
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
            for model_type in ["tcn", "mlp"]:
                for emb_at_ith_layer in ["first", "all"]:
                    for sim_measure_type in [
                        "dtw",
                    ]:
                        print(
                            dataset,
                            model_type,
                            embedding_type,
                            emb_at_ith_layer,
                        )
                        if sim_measure_type == "mp" and "2017" not in dataset:
                            continue

                        if full_bayes and embedding_type == EmbeddingType.Bayes:
                            continue

                        if model_type == "mlp" and emb_at_ith_layer != "first":
                            continue

                        if (
                            emb_at_ith_layer == "none"
                            and embedding_type == EmbeddingType.Bayes
                        ):
                            continue

                        year, data_type = get_data_type_and_year(
                            data_folder, args.gefcom_type
                        )

                        result_folder = str_to_path(
                            args.result_folder + f"/{year}_{data_type}"
                        )
                        str_to_path(str(result_folder) + "/preds", create_folder=True)

                        with open(f"{result_folder}/splits.json") as f:
                            splits = json.load(f)

                        relevant_files = L()
                        for cur_run_id in splits.keys():
                            with open(
                                result_folder
                                / f"park_id_to_name_{cur_run_id}_{model_type}.json"
                            ) as input_file:
                                park_id_to_name = json.load(input_file)
                                adapt_dict_path_to_backend(park_id_to_name, SERVER)
                                relevant_files += park_id_to_name.values()

                        all_files = list(np.unique(relevant_files))

                        if is_mlp(model_type):
                            config = get_mlp_config(year, data_type, embedding_type)
                        else:
                            config = get_cnn_config(
                                year, data_type, embedding_type, emb_at_ith_layer
                            )

                        adapt_dict_path_to_backend(splits, SERVER)

                        str_to_path(
                            str(result_folder) + "/zero_shot", create_folder=True
                        )
                        most_similar_parks_file = (
                            result_folder
                            / f"zero_shot/most_similar_parks_{data_type}_{year}_{sim_measure_type}.pkl"
                        )

                        if most_similar_parks_file.exists():
                            most_similar_parks = pkl.load(
                                open(most_similar_parks_file, "rb")
                            )
                        else:
                            most_similar_parks = get_sim_matrix(
                                all_files, sim_measure_type
                            )

                            pkl.dump(
                                most_similar_parks,
                                open(most_similar_parks_file, "wb"),
                            )

                        dfs_zero_shot_res = []
                        for cur_run_id in tqdm.tqdm(preselect_keys(splits, run_id)):

                            cur_source_files = splits[cur_run_id]

                            target_files = get_target_files(
                                all_files, cur_source_files, data_type
                            )

                            source_model_file = (
                                result_folder
                                / f"{model_type}_source_model_{full_bayes}_{cur_run_id}_{embedding_type}_residuals_{emb_at_ith_layer}.pkl"
                            ).absolute()

                            df_zero_shot_res = create_results(
                                target_files,
                                source_model_file,
                                result_folder,
                                cur_run_id,
                                model_type,
                                full_bayes,
                                emb_at_ith_layer,
                                year,
                                most_similar_parks,
                                sim_measure_type,
                            )
                            df_zero_shot_res["RunID"] = cur_run_id
                            dfs_zero_shot_res.append(df_zero_shot_res)

                        dfs_zero_shot_res = pd.concat(
                            dfs_zero_shot_res, axis=0
                        ).reset_index()
                        print(dfs_zero_shot_res.describe())

                        dfs_zero_shot_res.to_csv(
                            result_folder
                            / f"zero_shot/{model_type}__fullBayes_{full_bayes}_embType_{embedding_type}_residual_{emb_at_ith_layer}_simMeasure_{sim_measure_type}_zero_shot_results.csv",
                            sep=";",
                        )
