import os
import argparse
import types
from multiprocessing import Pool
from fastcore.basics import listify
from fastcore.imports import isinstance_str

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error as sk_mse
from ray import tune
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from fastai.data.transforms import IndexSplitter, RandomSplitter
from fastai.data.core import DataLoaders
from fastai.torch_core import to_np
from fastcore.utils import Path
from fastcore.foundation import L
import torch

from dies.data import split_by_n_weeks
from dies.data import (
    TimeseriesTransform,
    split_by_year,
    get_samples_per_day,
    create_consistent_number_of_sampler_per_day,
    tp_from_df_from_dtypes,
)
from rep.shift_features import ShiftFeatures


def add_lead_time(df):
    n_samples_per_day = get_samples_per_day(df)
    lead_times = np.concatenate(
        [
            np.arange(0, n_samples_per_day)
            for _ in range(int(df.shape[0] / n_samples_per_day))
        ]
    )
    df["LeadTime"] = lead_times.astype(np.int32)


def get_all_dfs(
    files: Path,
    include_shift_features: bool,
    standardize: bool,
    old_drop_cols: bool = False,
):
    files = listify(files)
    train_dfs, test_dfs = L(), L()
    non_empty_files = L()

    if files is None or len(files) == 0:
        None, None, None

    df_list = L(read_file(f, old_drop_cols=old_drop_cols) for f in files)

    for idx, cur_df in enumerate(df_list):
        file = files[idx]
        df_train, df_test = cur_df

        if isinstance(df_train, type(None)) or len(df_train) == 0 or len(df_test) == 0:
            continue

        non_empty_files += file
        df_train["TaskID"] = idx
        df_test["TaskID"] = idx
        n_samples_per_day = get_samples_per_day(df_train)
        create_consistent_number_of_sampler_per_day(df_train, n_samples_per_day)
        create_consistent_number_of_sampler_per_day(df_test, n_samples_per_day)
        add_lead_time(df_train)
        add_lead_time(df_test)

        if standardize:
            for c in df_train.columns:
                if (
                    c != "PowerGeneration"
                    and c != "TaskID"
                    and c != "Hour"
                    and c != "Month"
                    and c != "LeadTime"
                ):
                    scaler = StandardScaler()
                    df_train[c] = scaler.fit_transform(df_train[[c]])
                    df_test[c] = scaler.transform(df_test[[c]])

        train_dfs += df_train
        test_dfs += df_test

    if len(train_dfs) > 0 or len(test_dfs) > 0:
        return (
            pd.concat(train_dfs, axis=0),
            pd.concat(test_dfs, axis=0),
            non_empty_files,
        )
    else:
        return None, None, None


def get_n_largest_files(files: Path, n: int):
    sisez = L()
    for f in files:
        df = read_file(f, do_prep=False)
        sisez += df.shape[0]

    return files[np.argsort(sisez)[:n]]


def create_timestampindex(df: pd.DataFrame) -> pd.DataFrame:
    open_dataset = False
    if df.index.name == "TimeUTC":
        return df

    df.reset_index(drop=True, inplace=True)
    if "TimeUTC" not in df.columns and "PredictionTimeUTC" in df.columns:
        df = df.rename({"PredictionTimeUTC": "TimeUTC"}, axis=1)
    if "0000-" in df.TimeUTC[0]:
        #  as the date is anonymoused crete a proper date
        df.TimeUTC = df.TimeUTC.apply(
            lambda x: x.replace("0000-", "2015-").replace("0001-", "2016-")
        )
        open_dataset = True
    df.TimeUTC = pd.to_datetime(df.TimeUTC, infer_datetime_format=True, utc=True)
    df.set_index("TimeUTC", inplace=True)
    df.index = df.index.rename("TimeUTC")

    #  for GermanSolarFarm, the index is not corret. Should have a three hour resolution but is one...
    if "SolarRadiationDirect" in df.columns and open_dataset:
        i, new_index = 0, []
        for cur_index in df.index:
            new_index.append(cur_index + pd.DateOffset(hours=i))
            i += 2
        df.index = new_index

    return df


def assure_consisten_samples_per_day(df: pd.DataFrame) -> pd.DataFrame:
    n_samples = get_samples_per_day(df)
    return create_consistent_number_of_sampler_per_day(
        df, num_samples_per_day=n_samples
    )


def normalize_and_check_power(df: pd.DataFrame) -> pd.DataFrame:
    if "MaxPowerGeneration" in df.columns:
        df.PowerGeneration = df.PowerGeneration.apply(
            float
        ) / df.MaxPowerGeneration.apply(float)
        df.drop("MaxPowerGeneration", axis=1, inplace=True)

    df = df[df.PowerGeneration < 1.1]

    return df


def add_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = df.index.month
    df["Hour"] = df.index.hour

    return df


def drop_columns(df: pd.DataFrame, old_drop_cols: bool = False) -> pd.DataFrame:
    cols_to_drop = [
        c
        for c in df.columns
        if ("sin" in c.lower() and "wind" not in c.lower())
        or ("cos" in c.lower() and "wind" not in c.lower())
        or "t_minus" in c.lower()
        or "t_plus" in c.lower()
        or "forecastingtime" in c.lower()
        or "season" in c.lower()
    ]
    if old_drop_cols:
        cols_to_drop = [
            c
            for c in df.columns
            if ("sin" in c.lower())
            or ("cos" in c.lower())
            or "t_minus" in c.lower()
            or "t_plus" in c.lower()
            or "forecastingtime" in c.lower()
            or "season" in c.lower()
        ]
        DeprecationWarning(
            "Old columns to remove approach. Will be removed in the future."
        )

    df = df.drop(cols_to_drop, axis=1)

    return df


def split_in_train_and_test(df: pd.DataFrame):
    if "TestFlag" not in df.columns:
        year = df.index[0].year
        return split_by_year(df, year + 1)
    else:
        df_train = df[df.TestFlag == 0]
        df_test = df[df.TestFlag == 1]

        return df_train, df_test


def most_relevant_feature(df: pd.DataFrame) -> str:
    most_relevant_feature = ""
    if "WindSpeed100m" in df.columns:
        most_relevant_feature = "WindSpeed100m"
    elif "WindSpeed122m" in df.columns:
        most_relevant_feature = "WindSpeed122m"
    elif "SolarRadiationDirect" in df.columns:
        most_relevant_feature = "SolarRadiationDirect"
    elif "DirectRadiation" in df.columns:
        most_relevant_feature = "DirectRadiation"
    elif "NetSolarRad" in df.columns:
        most_relevant_feature = "NetSolarRad"
    else:
        raise ValueError(f"Could not find most relevant feature from {df.columns}")

    return most_relevant_feature


def shift_features(df: pd.DataFrame) -> pd.DataFrame:

    sw_feature = most_relevant_feature(df)

    df = ShiftFeatures(
        sw_feature,
        eval_position="past",
        step_sizes=[1, 2, 3],
    ).fit_transform(df)

    return df


def read_file(
    file: Path,
    do_prep: bool = True,
    include_shift_features: bool = False,
    old_drop_cols: bool = False,
) -> pd.DataFrame:
    if isinstance(file, str):
        file = str_to_path(file, create_folder=False)

    if file.suffix == ".h5":
        df = read_hdf(file)
    elif file.suffix == ".csv":
        df = read_csv(file)
    else:
        raise f"File ending of file {file} not supported."

    if len(df) == 0 or ("TimeUTC" not in df.columns and "TimeUTC" not in df.index.name):
        return None, None

    if do_prep:
        df = drop_columns(df, old_drop_cols=old_drop_cols)
        df = handle_nas(df)
        df = create_timestampindex(df)

        df = interpolate(df)

        df = normalize_and_check_power(df)
        df = add_categoricals(df)
        if include_shift_features:
            df = shift_features(df)
        df_train, df_test = split_in_train_and_test(df)

        df_train, df_test = (
            assure_consisten_samples_per_day(df_train),
            assure_consisten_samples_per_day(df_test),
        )

        return df_train, df_test
    else:
        return df


def interpolate(df: pd.DataFrame):
    n_samples = get_samples_per_day(df)
    if n_samples == 24:
        sample_time = "15Min"
    else:
        sample_time = "1H"

    df = df[~df.index.duplicated()]
    upsampled = df.resample(sample_time)
    df = upsampled.interpolate(method="linear", limit=5)

    if "Hour" in df.columns:
        df["Hour"] = df.index.hour
    if "Month" in df.columns:
        df["Month"] = df.index.month
    if "Day" in df.columns:
        df["Day"] = df.index.day
    if "Week" in df.columns:
        df["Week"] = df.index.week

    return df


def handle_nas(df: pd.DataFrame):
    df = df.fillna(df.mean())
    df = df.dropna(axis=1)
    return df


def read_hdf(file: Path, key: str = "/powerdata"):
    with pd.HDFStore(file, "r") as store:
        if key in store.keys():
            df = store[key]
        else:
            df = pd.DataFrame()

    return df


def read_csv(file: Path):
    df = pd.read_csv(str(file), sep=";")

    return df


def str_to_path(folder: str, create_folder=True):
    if "~" in folder:
        folder = os.path.expanduser(folder)

    folder = Path(folder)
    if create_folder:
        folder.mkdir(parents=True, exist_ok=True)

    return folder


def split_by_n_weeks_by_group(dfs, group_by="TaskID", every_n_weeks=6, for_n_weeks=1):

    dfs_train, dfs_valid = [], []
    for k, df in dfs.groupby(group_by):
        train_mask, valid_mask = split_by_n_weeks(
            df, every_n_weeks=every_n_weeks, for_n_weeks=for_n_weeks
        )
        dfs_train.append(df[train_mask])
        dfs_valid.append(df[valid_mask])

    dfs_train = pd.concat(dfs_train, axis=0)
    dfs_valid = pd.concat(dfs_valid, axis=0)

    return dfs_train, dfs_valid


def add_args(parser):
    parser.add_argument(
        "--data_folder",
        help="Path to the data prepared data folder.",
        # default="~/data/prophesy-data/WindSandbox2015/",
        # default="~/data/prophesy-data/PVSandbox2015/",
        default="~/data/prophesy-data/GEFCOM2014/",
    )

    parser.add_argument(
        "--gefcom_type",
        help="wind or pv.",
        default="pv",
    )

    parser.add_argument(
        "--result_folder",
        help="Path to the store results.",
        default="./results/",
    )

    parser.add_argument(
        "--num_samples",
        help="Number of samples in random search.",
        default=1,
    )

    parser.add_argument(
        "--n_files",
        help="Number of source_files_to_use.",
        default=3,
    )

    parser.add_argument(
        "--embedding_type",
        help="Bayes or normal embedding. [bayes/normal]",
        default="bayes",
    )

    parser.add_argument(
        "--debug",
        help="Flag for debugging.",
        default=1,
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
        help="Whether to run all or 0/1/...",
        default="all",
    )


def get_error(
    learn,
    ds_idx=1,
    test_dl=None,
    return_forecasts=False,
    full_bayes=False,
    n_bayes_samples=50,
):
    if test_dl is not None:
        preds, targets = learn.get_preds(dl=test_dl)
        if full_bayes:
            preds = [preds]
            for i in range(0, n_bayes_samples):
                preds_i, _ = learn.get_preds(dl=test_dl)
                preds.append(preds_i)
            preds = torch.cat(preds, 1).mean(1).unsqueeze(1)

    else:
        preds, targets = learn.get_preds(ds_idx=ds_idx)
        if full_bayes:
            preds = [preds]
            for i in range(0, n_bayes_samples):
                preds_i, _ = learn.get_preds(dl=test_dl)
                preds.append(preds_i)
            preds = torch.cat(preds, 1).mean(1).unsqueeze(1)

    preds, targets = to_np(preds).reshape(-1), to_np(targets).reshape(-1)

    preds[preds < 0] = 0
    preds[preds > 1.1] = 1.1

    rmse_value = sk_mse(targets, preds) ** 0.5
    if return_forecasts:
        return rmse_value, preds
    else:
        return rmse_value


def dataframe_to_tp_dataloader(
    config, df_train, standardize, save_model, n_folds=3, add_y_to_x=False
):
    if not config["keep_month"]:
        df_train.drop("Month", axis=1, inplace=True)

    if save_model:
        tp = tp_from_df_from_dtypes(
            df_train,
            "PowerGeneration",
            valid_percent=0.1,
            standardize_X=standardize,
            do_split_by_n_weeks=False,
            add_y_to_x=add_y_to_x,
            # TODO: does this make sense?
            categorify=False,
        )

        dls = L(tp.dataloaders(bs=config["batch_size"]))
    else:
        n_samples = df_train.shape[0]
        dls = L()
        kf = KFold(n_splits=n_folds)
        for _, val_idx in kf.split(range(n_samples)):

            tp = tp_from_df_from_dtypes(
                df_train,
                "PowerGeneration",
                val_idxs=val_idx,
                standardize_X=standardize,
                do_split_by_n_weeks=False,
                add_y_to_x=add_y_to_x,
            )

            dls += tp.dataloaders(bs=config["batch_size"])

    n_input_features = len(dls[0].train_ds.cont_names)

    return n_input_features, dls


def dataframe_to_ts_dataloader(
    df_train,
    n_samples_per_day,
    config,
    save_model,
    n_folds=3,
    add_y_to_x=False,
):
    standardize = False

    df_train.drop("Hour", axis=1, inplace=True)
    if not config["keep_month"]:
        df_train.drop("Month", axis=1, inplace=True)

    if save_model:
        return L(
            _dataframe_to_ts_dataloader_after_training(
                df_train,
                n_samples_per_day,
                standardize,
                config["batch_size"],
                add_y_to_x=add_y_to_x,
            )
        )
    else:
        return dataframe_to_ts_dl_kfold(
            df_train,
            standardize,
            n_samples_per_day,
            n_folds,
            config["batch_size"],
            add_y_to_x=add_y_to_x,
        )


def dataframe_to_ts_dl_kfold(
    df_train, standardize, n_samples_per_day, n_folds, batch_size, add_y_to_x
):
    tp_train = tp_from_df_from_dtypes(
        df_train,
        "PowerGeneration",
        valid_percent=None,
        standardize_X=standardize,
        add_y_to_x=add_y_to_x,
    )

    train_source_tl = TimeseriesTransform(
        tp_train,
        timeseries_length=n_samples_per_day,
        batch_first=True,
        sequence_last=True,
        is_train=False,
        drop_inconsistent_cats=True,
    )

    num_samples = len(train_source_tl)
    dls_trains = L()
    kf = KFold(n_splits=n_folds)
    for _, val_idx in kf.split(np.arange(num_samples)):

        dls_train = train_source_tl.to_dataloaders(
            bs=batch_size,
            shuffle=True,
            splits=IndexSplitter(valid_idx=val_idx),
            drop_last=True,
        )
        dls_trains += dls_train

    return dls_trains


def _dataframe_to_ts_dataloader_after_training(
    df_train, n_samples_per_day, standardize, batch_size, add_y_to_x=False
):
    df_train = create_consistent_number_of_sampler_per_day(df_train, n_samples_per_day)

    tp_train = tp_from_df_from_dtypes(
        df_train,
        "PowerGeneration",
        valid_percent=None,
        standardize_X=standardize,
        add_y_to_x=add_y_to_x,
    )

    train_source_tl = TimeseriesTransform(
        tp_train,
        timeseries_length=n_samples_per_day,
        batch_first=True,
        sequence_last=True,
        is_train=False,
        drop_inconsistent_cats=True,
    )

    tls = train_source_tl.to_tfmd_lists(splits=RandomSplitter(valid_pct=0.1))
    dls = tls.dataloaders(bs=batch_size)

    return dls
