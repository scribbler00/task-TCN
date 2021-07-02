# %%

from confer.utils_eval import (
    add_naming_columns,
    get_baseline,
    get_results_as_df,
    t_test,
    wilcoxon_test,
    zero_shot_error_hyphothesis_test,
)
import glob
from dies.embedding import EmbeddingType
import pandas as pd
from fastcore.foundation import L
from fastcore.foundation import Path
from confer.utils_eval import bold_extreme_values, df_to_latex, filter_models_eval
import numpy as np


# %%


def get_mtl_error_results(base_folder, data_type):
    baseline_name = (
        "mlp__fullBayes_False_embType_EmbeddingType.Bayes_residual_first_mtl_results"
    )
    sign_symbol = "*"
    dfs, files = get_results_as_df(base_folder, filter_str="")

    df_baseline, baseline_id = get_baseline(dfs, files, baseline_name)
    df_baseline = df_baseline.groupby("Files").mean()
    print("baseline_id", baseline_id, files[baseline_id])

    sigs = zero_shot_error_hyphothesis_test(dfs, files, baseline_id, sign_symbol)
    df_summary = L()

    for idx, df in enumerate(dfs):
        print(files[idx].stem)
        df_std = df.groupby("Files").std()
        df = df.groupby("Files").mean()
        df_ss = pd.merge(
            df_baseline,
            df.groupby("Files").mean(),
            on="Files",
            suffixes=("Baseline", "Reference"),
        )
        skill = np.mean(1 - df_ss.ZeroShotErrorReference / df_ss.ZeroShotErrorBaseline)

        df = df.mean().T.to_frame()
        df.columns = ["nRMSE"]

        df.insert(1, "Skill", skill)
        add_naming_columns(df, files[idx].stem)
        df["std"] = df_std.mean()
        df_summary += df

    df_summary = pd.concat(df_summary)
    df_summary["DataType"] = (
        data_type.upper() if data_type == "pv" else data_type.capitalize()
    )

    df_summary = df_summary.set_index(["Model", "Type", "EmbPos", "DataType"])

    for col in df_summary.columns:
        df_summary[col] = bold_extreme_values(
            df_summary[col],
            format_string="%.3f",
            max_=True if col == "Skill" else False,
        )

    df_summary["nRMSE"] = df_summary["nRMSE"] + sigs
    df_summary = df_summary.sort_index()

    return df_summary


# %%
base_folder = "../results/2015_pv/mtl/"
base_folder = "/mnt/1B5D7DBC354FB150/res_task_tcn/2015_pv/mtl/"
df_summary_pv = get_mtl_error_results(
    base_folder,
    "pv",
)

base_folder = "../results/2015_wind/mtl/"
base_folder = "/mnt/1B5D7DBC354FB150/res_task_tcn/2015_wind/mtl/"
df_summary_wind = get_mtl_error_results(base_folder, "wind")

# %%
df_summary = (
    pd.concat([df_summary_pv, df_summary_wind])
    .unstack(3)
    .swaplevel(0, axis=1)
    .sort_index(axis=1)
)

new_level_values = df_summary.index.get_level_values(-1) + [
    " (Ref.)" if i != 0 else " (BS)" for i in range(len(df_summary))
]
df_summary.index = df_summary.index.droplevel(-1)
df_summary.insert(0, "EmbPos", new_level_values)
df_summary = df_summary.reset_index().set_index(["Model", "Type", "EmbPos"])


df_summary

# %%
print(
    "\\begin{center}\n"
    + df_to_latex(df_summary).replace("begin{tabular}", "begin{tabular}[tb]")
    + "\end{center}"
)

# %%
