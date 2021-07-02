import os
from fastcore.basics import listify
from fastai.tabular.core import TabularPandas, TabDataLoader
import pandas as pd
from pathlib import PosixPath
from sklearn.metrics import mean_squared_error as sk_mse
from fastai.torch_core import to_np, tensor
from fastcore.utils import Path
from fastcore.foundation import L
import torch
from dies.data import get_samples_per_day
from dies.data import (
    get_samples_per_day,
)
from dies.embedding import EmbeddingType


def read_csv(file: PosixPath):
    df = pd.read_csv(str(file), sep=";")

    return df


def str_to_path(folder: str, create_folder=True):
    if "~" in folder:
        folder = os.path.expanduser(folder)

    folder = Path(folder)
    if create_folder:
        folder.mkdir(parents=True, exist_ok=True)

    return folder


def add_args(parser):
    parser.add_argument(
        "--data_folder",
        help="Path to the data prepared data folder.",
        # default="~/data/prophesy-data/WindSandbox2015/",
        default="~/data/prophesy-data/PVSandbox2015/",
        # default="~/data/prophesy-data/WindSandbox2017/",
        # default="~/data/prophesy-data/PVSandbox2017/",
        # default="~/data/prophesy-data/GEFCOM2014/",
    )

    parser.add_argument(
        "--gefcom_type",
        help="wind or pv.",
        default="pv",
    )

    parser.add_argument(
        "--result_folder",
        help="Path to the store results.",
        # default="/home/scribbler/workspace/convolutional_transfer/results/",
        default="/mnt/1B5D7DBC354FB150/res_task_tcn/",
    )

    parser.add_argument(
        "--embedding_type",
        help="Bayes or normal embedding. [bayes/normal]",
        default="bayes",
    )

    parser.add_argument(
        "--finetune",
        help="whether to fine tune the model or not [yes/no]",
        default="yes",
    )

    parser.add_argument(
        "--full_bayes",
        help="whether to use a full bayesian network [yes/no]",
        default="no",
    )

    parser.add_argument(
        "--run_id",
        help="Whether to run all or 0/1/... Only used for target.",
        default="all",
    )


def get_error(
    learn,
    ds_idx=1,
    test_dl=None,
    return_forecasts=False,
    full_bayes=False,
    n_bayes_samples=50,
    device="cpu",
):
    learn.model.eval()

    cur_dl = learn.dls.train_ds
    if test_dl is not None:
        cur_dl = test_dl
    elif ds_idx == 1:
        cur_dl = learn.dls.valid_ds
    if type(cur_dl) == TabularPandas or type(cur_dl) == TabDataLoader:
        cats = tensor(
            cur_dl.cats.values,
        ).long()
        xs = tensor(cur_dl.conts.values)
        targets = tensor(cur_dl.y.values)
    else:
        cats = cur_dl.cats
        xs = cur_dl.xs
        targets = cur_dl.ys

    with torch.no_grad():
        preds = learn.model(cats.to(device), xs.to(device))

    if full_bayes:
        raise NotImplementedError

    preds, targets = to_np(preds).reshape(-1), to_np(targets).reshape(-1)

    preds[preds < 0] = 0
    preds[preds > 1.1] = 1.1

    rmse_value = sk_mse(targets, preds) ** 0.5
    if return_forecasts:
        return rmse_value, preds
    else:
        return rmse_value


def adapt_dict_path_to_backend(dict_to_transform, server):
    match_pattern = "/mnt/work/prophesy/"
    replace_pattern = "~/data/"

    if server:
        match_pattern = "/home/scribbler/data"
        replace_pattern = "/mnt/work/prophesy/"

    for k in dict_to_transform:
        cur_element = dict_to_transform[k]
        cur_element = listify(cur_element)

        cur_element = [
            str_to_path(f.replace(match_pattern, replace_pattern), create_folder=False)
            for f in cur_element
        ]

        dict_to_transform[k] = cur_element


def get_source_preds(
    base_folder,
    target_file,
    source_model_file,
):
    preds_target_file = (
        f"{base_folder}/preds_source/{target_file.stem}_{source_model_file.stem}.csv"
    )
    preds_target_file = Path(preds_target_file)

    df_source = None
    if preds_target_file.exists():
        df_source = pd.read_csv(preds_target_file, sep=";")
        df_source.rename({"Unnamed: 0": "TimeUTC"}, axis=1, inplace=True)
        df_source.TimeUTC = pd.to_datetime(
            df_source.TimeUTC, utc=True, infer_datetime_format=True
        )

    return df_source


def filter_source_preds(
    df_source,
    task_id,
    num_days_training,
    split_date,
    months_to_train_on,
):
    mask = df_source.TaskID == task_id

    df_source = df_source[mask].reset_index()

    mask = df_source.TimeUTC < pd.to_datetime(
        split_date, utc=True, infer_datetime_format=True
    )
    df_source = df_source[mask]
    df_source.set_index("TimeUTC", inplace=True)

    n_samples_per_day = get_samples_per_day(df_source)

    mask = df_source.index.month.isin(months_to_train_on)

    df_source = df_source[mask]

    if len(df_source) > 0:
        df_source = df_source[-(num_days_training * n_samples_per_day) :]

    return df_source


def get_data_type_and_year(data_folder, gefcom_type):
    year = "2015" if "2015" in str(data_folder) else "2014"
    year = "2017" if "2017" in str(data_folder) else year

    if "wind" in str(data_folder).lower():
        data_type = "wind"
    elif "gefcom" in str(data_folder).lower():
        data_type = gefcom_type
    else:
        data_type = "pv"

    return year, data_type


def is_gefcom(data_folder):
    return "gefcom" in str(data_folder).lower()


def get_target_files(all_files, source_files, data_type):
    target_files = [
        f
        for f in all_files
        if ((f.suffix == ".h5" or f.suffix == ".csv") and f not in source_files)
    ]

    blacklist = get_blacklist()
    for target_file in target_files:
        if target_file.stem in blacklist:
            print("Skipped:", target_file.stem)
            continue

    if is_gefcom(str(all_files[0])):
        target_files = [
            f
            for f in target_files
            if (data_type == "pv" and "solar" in str(f).lower())
            or (data_type == "wind" and "solar" not in str(f).lower())
        ]

    return target_files


def is_full_bayes(full_bayes):
    return True if full_bayes.lower() == "yes" else False


def preselect_keys(splits, run_id):
    run_id_keys_all = list(splits.keys())
    if run_id != "all":
        run_id = int(run_id)
        run_id_keys_all = [run_id_keys_all[run_id]]

    return run_id_keys_all


def get_embedding(embedding_name):
    if "bayes" in embedding_name:
        embedding_type = EmbeddingType.Bayes
    elif "normal" in embedding_name:
        embedding_type = EmbeddingType.Normal
    else:
        raise ValueError

    return embedding_type


def get_split_year(dataset_year):
    dataset_year = int(dataset_year)
    if dataset_year == 2015:
        return 2016
    elif dataset_year == 2017:
        return 2016
    elif dataset_year == 2014:
        return 2015
    else:
        raise NotImplementedError


def is_mlp(name):
    return "mlp" in str(name).lower()


def match_time_stamps(df_1, df_2):
    mask_df1 = df_1.index.isin(df_2.index)
    df_1 = df_1[mask_df1]

    mask_df2 = df_2.index.isin(df_1.index)
    df_2 = df_2[mask_df2]

    return df_1, df_2


def get_blacklist():
    blacklist_wind = [""]
    blacklist_pv = [""]
    return blacklist_wind + blacklist_pv