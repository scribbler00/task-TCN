# %%
import collections
import glob
from typing import DefaultDict
from fastcore.foundation import L
from fastcore.foundation import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager
import matplotlib.pyplot as plt
from confer.utils_eval import (
    correct_types_inductive_tl,
    create_boxplot_inductive_tl_grouped_by_day,
    create_inductive_tl_multicolumn_nrmse_table,
    create_inductive_tl_multicolumn_skill_table,
    df_to_latex_file,
    filter_tl_hyperparameter_search,
    get_baseline,
    get_results_as_df,
    highlight_inductive_tl_results,
    prepare_inductive_tl_results,
    season_id_to_name,
)

# %%
sns.set()
sns.color_palette("colorblind", 8)
# sns.set_context("poster")
sns.set_style("whitegrid")
mpl.rcParams["legend.loc"] = "upper right"
FONT_SIZE = 40
params = {
    "axes.labelsize": FONT_SIZE,  # fontsize for x and y labels (was 10)
    "axes.titlesize": FONT_SIZE,
    "font.size": FONT_SIZE,  # was 10
    "legend.fontsize": FONT_SIZE,  # was 10
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "font.family": "Times New Roman",
}
mpl.rcParams.update(params)


# %%
baseline_name = "mlp_fullBayes_False_embType_EmbeddingType.Bayes_residual_first_embAdaption_reset_ftType_emb_only_results"

base_folder = "../results/2015_pv/tl/"
base_folder = "/mnt/1B5D7DBC354FB150/res_task_tcn/2015_pv/tl/"
data_type = "pv" if "pv" in base_folder else "wind"
dfs, rel_files = get_results_as_df(base_folder)
dfs = L(filter_tl_hyperparameter_search(df) for df in dfs)
dfs = L(correct_types_inductive_tl(df) for df in dfs)
df_baseline, baseline_id = get_baseline(dfs, rel_files, baseline_name)
sign_symbol = "*"
# %%
create_boxplot_inductive_tl_grouped_by_day(
    data_type, dfs, rel_files, df_baseline, baseline_id, legend_font_size=FONT_SIZE - 10
)

# %%


all_results = prepare_inductive_tl_results(
    data_type, dfs, rel_files, df_baseline, baseline_id
)


# %%
# num_decimals = 3

# for cur_season in [0, 1, 2, 3, 4]:
#     print(cur_season)

#     df_all_results = pd.concat(all_results)

#     df_all_results_single_season_days_training = highlight_inductive_tl_results(
#         num_decimals, cur_season, df_all_results
#     )

#     df_results_nrmse = create_inductive_tl_multicolumn_nrmse_table(
#         df_all_results_single_season_days_training, num_decimals
#     )
#     new_level_values = df_results_nrmse.index.get_level_values(-1) + [
#         " (Ref.)" if i != 1 else " (BS)" for i in range(len(df_results_nrmse))
#     ]
#     df_results_nrmse.index = df_results_nrmse.index.droplevel(-1)
#     df_results_nrmse.insert(0, "Init", new_level_values)
#     df_results_nrmse = df_results_nrmse.reset_index().set_index(
#         ["Model", "Type", "EmbPos", "Init"]
#     )

#     df_all_results_single_season_days_training = highlight_inductive_tl_results(
#         num_decimals,
#         cur_season,
#         df_all_results,
#     )
#     df_results_skill = create_inductive_tl_multicolumn_skill_table(
#         df_all_results_single_season_days_training, num_decimals
#     )
#     new_level_values = df_results_skill.index.get_level_values(-1) + [
#         " (Ref.)" if i != 1 else " (BS)" for i in range(len(df_results_skill))
#     ]
#     df_results_skill.index = df_results_skill.index.droplevel(-1)
#     df_results_skill.insert(0, "Init", new_level_values)
#     df_results_skill = df_results_skill.reset_index().set_index(
#         ["Model", "Type", "EmbPos", "Init"]
#     )

#     eval_type = "nRMSE"
#     file_name = f"../doc/figs/itl/{data_type}_{eval_type}_table_by_days_for_{season_id_to_name(cur_season)}.tex"
#     season_name = season_id_to_name(cur_season)
#     sign_symbol
#     caption = (
#         f"Mean {eval_type} for {season_name} of {data_type} data set. Siginificant diffference of the reference (Ref.) compared to the baseline (BS) is tested through the Wilcoxon signed-rank test with "
#         + "$\\alpha=0.05$ "
#         + f"and marked with {sign_symbol}."
#     )
#     df_to_latex_file(df_results_nrmse, file_name, caption=caption)

#     eval_type = "skill"
#     file_name = f"../doc/figs/itl/{data_type}_{eval_type}_table_by_days_for_{season_id_to_name(cur_season)}.tex"
#     season_name = season_id_to_name(cur_season)
#     sign_symbol
#     caption = (
#         f"Mean {eval_type} for {season_name} of {data_type} data set. Siginificant diffference of the reference (Ref.) compared to the baseline (BS) is tested through the Wilcoxon signed-rank test with "
#         + "$\\alpha=0.05$ "
#         + f"and marked with {sign_symbol}."
#     )
#     df_to_latex_file(df_results_skill, file_name, caption=caption)


# %%


# %%
