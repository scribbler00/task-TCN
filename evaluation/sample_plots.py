# %%
from confer.utils import str_to_path
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from rep.utils_data import SourceData

sns.set()
sns.color_palette("colorblind", 8)
# sns.set_context("poster")
sns.set_style("whitegrid")
mpl.rcParams["legend.loc"] = "upper right"
FONT_SIZE = 45
params = {
    "axes.labelsize": FONT_SIZE,  # fontsize for x and y labels (was 10)
    "axes.titlesize": FONT_SIZE,
    "font.size": FONT_SIZE,  # was 10
    "legend.fontsize": FONT_SIZE - 3,  # was 10
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "font.family": "Times New Roman",
}
mpl.rcParams.update(params)
# %%

# %%
def filter(files, model_type, init, emb_pos="first"):
    return [
        f
        for f in files
        if model_type in f
        and "embType_bayes_" in f
        and "numDays_30" in f
        and init in f
        and park_name in f
        and "curSeason_1" in f
        and emb_pos in f
    ]


def read_df(f):
    df = pd.read_csv(f, sep=";")
    df.columns = ["TimeUTC", "PowerGeneration", "Preds"]
    df.TimeUTC = pd.to_datetime(df.TimeUTC, infer_datetime_format=True, utc=True)
    df.set_index("TimeUTC", inplace=True)
    return df


def plot(df_nwp, df_tcn, df_mlp, nwp_col, name, length=1000, start=6000):
    plt.figure(figsize=(16, 11))

    end = start + length
    plt.plot(
        df_tcn.index[start:end],
        df_tcn[start:end].PowerGeneration,
        label="Power",
        linestyle="solid",
    )
    plt.plot(
        df_tcn.index[start:end],
        df_tcn[start:end].Preds,
        label="Task-TCN",
        linestyle="dotted",
    )
    plt.plot(
        df_mlp.index[start:end],
        df_mlp[start:end].Preds,
        label="Baseline",
        linestyle="dashed",
    )
    plt.plot(
        df_nwp.index[start:end],
        df_nwp[start:end][nwp_col],
        label=nwp_col.replace("Direct", ""),
        linestyle="dashdot",
    )
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.xlabel("Day")
    plt.xticks(rotation=20)
    plt.ylabel("Normalized Power")
    plt.tight_layout()
    plt.savefig(f"./doc/figs/{name}.pdf")


# %%
base_folder = "../results/2015_wind/preds/"
base_folder = "/mnt/1B5D7DBC354FB150/res_task_tcn/2015_wind/preds/"
data_folder = "~/data/prophesy-data/WindSandbox2015/"
files = glob.glob(base_folder + "/*.csv")

park_name = "wf26"
df_nwp = SourceData(
    [str_to_path(data_folder + park_name + ".h5", create_folder=False)],
    standardize=False,
).df_test

# %%
mlp_files = filter(files, "mlp", "reset")
tcn_files = filter(files, "tcn", "copy", emb_pos="all")

df_tcn = read_df(tcn_files[0])
df_mlp = read_df(mlp_files[0])

plot(
    df_nwp,
    df_tcn,
    df_mlp,
    "WindSpeed100m",
    "sample_plot_ts_wind",
    length=500,
    start=5200,
)
# %%

base_folder = "../results/2015_pv/preds/"
base_folder = "/mnt/1B5D7DBC354FB150/res_task_tcn/2015_pv/preds/"
data_folder = "~/data/prophesy-data/PVSandbox2015/"
files = glob.glob(base_folder + "/*.csv")

park_name = "pv_05"
df_nwp = SourceData(
    [str_to_path(data_folder + park_name + ".h5", create_folder=False)],
    standardize=False,
).df_test

mlp_files = filter(files, "mlp", "reset")
tcn_files = filter(files, "tcn", "copy", emb_pos="all")

mlp_files, tcn_files
df_tcn = read_df(tcn_files[0])
df_mlp = read_df(mlp_files[0])

plot(df_nwp, df_tcn, df_mlp, "SolarRadiationDirect", "sample_plot_ts_pv", length=150)


# %%

# %%
