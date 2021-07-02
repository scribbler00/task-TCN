import glob
from pathlib import Path
from fastcore.foundation import L
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def t_test(
    baseline,
    reference,
    alpha_value=0.01,
):

    diff = baseline - reference

    _, p = stats.shapiro(diff)
    if p > alpha_value:

        _, p = stats.ttest_rel(baseline, reference)
        # print(model, p)
        if p < alpha_value:
            print("Model is significantly different to baseline.", p)
            return True
        else:
            print("Model is not significantly different to baseline", p)
            return False
    else:
        print("shapiro test failed", p)
        return False


def wilcoxon_test(baseline, reference, alpha_value=0.05, silent=True):

    _, p = stats.wilcoxon(baseline, reference, alternative="two-sided", mode="auto")

    if p < alpha_value:
        if not silent:
            print("Model is significantly different to baseline (reject H0).", p)
        return True
    else:
        if not silent:
            print(
                "Model is not significantly different to baseline (fail to reject H0).",
                p,
            )
        return False


def filter_tl_hyperparameter_search(cur_df):
    nRmses = L()
    keys = L()
    for k, df in cur_df.groupby(["Files", "Season", "DaysTraining"]):
        nRmse = df.sort_values("ValidationErrorsAfterFT").reset_index().loc[0]

        nRmses += nRmse.TestErrorsAfterFT
        keys += np.array(k)

    cur_df = pd.DataFrame(np.array(keys), columns=["Files", "Season", "DaysTraining"])
    cur_df["nRMSE"] = nRmses
    # cur_df.Season = cur_df.Season.astype(int)
    # cur_df.DaysTraining = cur_df.DaysTraining.astype(float)

    return cur_df


def format_column(data, format_string="%.3f", max_=False):
    formatted = data.apply(lambda x: format_string % x)
    return formatted


def bold_extreme_values(data, format_string="%.3f", max_=False):
    if max_:
        extrema = data != data.max()
    else:
        extrema = data != data.min()

    bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x: format_string % x)
    return formatted.where(extrema, bolded)


def df_to_latex(df):
    return df.to_latex(escape=False).replace("_", "\_").replace("%", "\%")


def df_to_latex_file(df):
    return df.to_latex(escape=False).replace("_", "\_").replace("%", "\%")


def filter_models_eval(files):
    files = L(
        f
        for f in files
        if (
            "fullBayes_True" not in f
            and "cnn" not in f
            #     # and ("EmbeddingType.Bayes" not in f or "none" in f.lower())
        )
    )

    return files


def get_results_as_df(base_folder, filter_str=""):

    files = glob.glob(base_folder + "*.csv")

    files = L(Path(f) for f in files if (filter_str in f))

    dfs = L()
    rel_files = L()
    for f in files:
        cur_df = pd.read_csv(f, sep=";").drop(
            ["Unnamed: 0", "level_0", "index", "SourceTaskID", "RunID"],
            axis=1,
            errors="ignore",
        )
        if len(cur_df) > 0:
            dfs += cur_df
            rel_files += f

    return dfs, rel_files


def get_baseline(dfs, files, baseline_name):
    baseline_id = -1
    for idx in range(len(dfs)):
        print(files[idx].stem)
        if files[idx].stem == baseline_name:
            baseline_id = idx
            break

    df_baseline = dfs[baseline_id]

    return df_baseline, baseline_id


def correct_types_inductive_tl(df):
    df.Season = df.Season.apply(lambda x: x.replace(".0", "")).astype(int)
    df.DaysTraining = df.DaysTraining.apply(lambda x: x.replace(".0", "")).astype(int)
    return df


def add_naming_columns(cur_df, cur_file):
    cur_df["Model"] = cur_file[0:3].upper()
    cur_df["Type"] = "Normal" if "normal" in cur_file.lower() else "Bayes"
    cur_df["EmbPos"] = "First" if "first" in cur_file.lower() else "All"
    return cur_df


def add_naming_columns_inductive_tl(cur_df, cur_file):
    cur_df = add_naming_columns(cur_df, cur_file)
    cur_df["Init"] = "Copy" if "copy" in cur_file.lower() else "Default"
    return cur_df


def zero_shot_error_hyphothesis_test(dfs, files, baseline_id, sign_symbol="*"):
    sigs = []
    for idx in range(len(dfs)):
        if idx == baseline_id:
            sigs.append(" ")
            continue
        df_t_test = pd.merge(
            dfs[baseline_id],
            dfs[idx],
            on="Files",
            suffixes=("Baseline", "Reference"),
        )
        if wilcoxon_test(
            df_t_test.ZeroShotErrorBaseline,
            df_t_test.ZeroShotErrorReference,
            alpha_value=0.05,
        ):
            sigs.append(sign_symbol)
        else:
            sigs.append("")
    return sigs


def season_id_to_name(id: int):
    season_id_to_name = {
        0: "winter",
        1: "spring",
        2: "summer",
        3: "autumn",
        4: "complete year",
    }
    return season_id_to_name[int(id)]


def create_boxplot_inductive_tl_grouped_by_day(
    data_type, dfs, rel_files, df_baseline, baseline_id, legend_font_size
):
    for season in [0, 1, 2, 3, 4]:
        all_season_dfs = L()
        for idx, cur_df in enumerate(dfs):
            cur_df = cur_df[cur_df.Season == season]
            cur_df = add_naming_columns_inductive_tl(cur_df, rel_files[idx].stem)
            if baseline_id == idx:
                cur_df["Init"] = cur_df["Init"] + "(BS)"
            all_season_dfs += cur_df
        all_season_dfs = pd.concat(all_season_dfs)
        all_season_dfs["ModelName"] = (
            all_season_dfs["Model"]
            + all_season_dfs["Type"]
            + all_season_dfs["EmbPos"]
            + all_season_dfs["Init"]
        )
        plt.figure(figsize=(16, 9))
        sns.boxplot(hue="DaysTraining", y="nRMSE", x="ModelName", data=all_season_dfs)
        if data_type == "pv":
            plt.ylim((0.05, 0.45))
        else:
            plt.ylim((0.05, 0.45))
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
            fontsize=legend_font_size,
            title_fontsize=legend_font_size,
            title="DaysTraining",
            fancybox=True,
        )
        # plt.title(
        #     f"Comparison for inductive TL experiment for {season_id_to_name(season)} of the {data_type} dataset.\n"
        #     + " BS marks the baseline."
        # )
        print(
            f"Results for inductive TL experiment for {season_id_to_name(season)} of the {data_type} dataset.\n"
            + " BS marks the baseline."
        )
        plt.xticks(rotation=90)
        plt.savefig(
            f"./doc/figs/itl/{data_type}_boxplot_by_days_for_{season_id_to_name(season)}.png",
            bbox_inches="tight",
        )


def highlight_inductive_tl_results(num_decimals, cur_season, df_all_results):
    df_all_results_single_season = df_all_results[df_all_results.Season == cur_season]
    df_all_results_single_season_days_training = L()
    for key, cur_df in df_all_results_single_season.groupby("DaysTraining"):
        cur_df["nRMSE"] = bold_extreme_values(
            cur_df["nRMSE"],
            max_=False,
            format_string=f"%.{num_decimals}f",
        )
        cur_df["StdnRMSE"] = format_column(
            cur_df["StdnRMSE"],
            max_=False,
            format_string=f"%.{num_decimals-1}f",
        )
        cur_df["Skill"] = bold_extreme_values(
            cur_df["Skill"],
            max_=max,
            format_string=f"%.{num_decimals}f",
        )
        cur_df["StdSkill"] = format_column(
            cur_df["StdSkill"],
            max_=False,
            format_string=f"%.{num_decimals-1}f",
        )
        df_all_results_single_season_days_training += cur_df
    df_all_results_single_season_days_training = pd.concat(
        df_all_results_single_season_days_training
    )
    return df_all_results_single_season_days_training


def create_inductive_tl_multicolumn_nrmse_table(
    df_all_results_single_season_days_training, num_decimals
):
    df_all_results_single_season_days_training["nRMSE"] = (
        df_all_results_single_season_days_training["nRMSE"].astype(str)
        # + "$^"
        + df_all_results_single_season_days_training["Sign"].astype(str)
        # + "$"
    )
    df_all_results_single_season_days_training.drop(
        ["Sign", "StdnRMSE", "Skill", "StdSkill"], inplace=True, axis=1
    )
    df_all_results_single_season_days_training = (
        df_all_results_single_season_days_training.set_index(
            ["Model", "Type", "EmbPos", "Init", "DaysTraining"]
        )
    )
    return df_all_results_single_season_days_training.drop(
        [
            "Season",
        ],
        axis=1,
    ).unstack()


def create_inductive_tl_multicolumn_skill_table(
    df_all_results_single_season_days_training, num_decimals
):
    df_all_results_single_season_days_training["Skill"] = (
        df_all_results_single_season_days_training["Skill"].astype(str)
        # + "$^"
        + df_all_results_single_season_days_training["Sign"].astype(str)
        # + "$"
    )
    df_all_results_single_season_days_training.drop(
        ["Sign", "StdnRMSE", "nRMSE", "StdSkill"], inplace=True, axis=1
    )
    df_all_results_single_season_days_training = (
        df_all_results_single_season_days_training.set_index(
            ["Model", "Type", "EmbPos", "Init", "DaysTraining"]
        )
    )
    return df_all_results_single_season_days_training.drop(
        [
            "Season",
        ],
        axis=1,
    ).unstack()


def prepare_inductive_tl_results(data_type, dfs, rel_files, df_baseline, baseline_id):
    sign_symbol = "*"
    all_results = L()
    for season in [0, 1, 2, 3, 4]:
        all_season_dfs = L()
        for idx, cur_df in enumerate(dfs):
            cur_df = add_naming_columns_inductive_tl(cur_df, rel_files[idx].stem)

            cur_df = cur_df[cur_df.Season == season]
            df_baseline_seasons = df_baseline[df_baseline.Season == season]

            for days_training, df_days_training in cur_df.groupby("DaysTraining"):

                df_baseline_days_training = df_baseline_seasons[
                    df_baseline_seasons.DaysTraining == days_training
                ]
                df_res_merged = pd.merge(
                    df_baseline_days_training,
                    df_days_training,
                    on=["Files"],
                    suffixes=("Baseline", "Reference"),
                )

                if idx != baseline_id:
                    if wilcoxon_test(
                        df_res_merged.nRMSEBaseline,
                        df_res_merged.nRMSEReference,
                        alpha_value=0.05,
                    ):
                        significant_sign = sign_symbol
                    else:
                        significant_sign = ""
                else:
                    significant_sign = ""
                df_res = pd.DataFrame(
                    {
                        "nRMSE": [df_days_training.nRMSE.mean()],
                        "DaysTraining": days_training,
                        "Season": season,
                        "StdnRMSE": [df_days_training.nRMSE.std()],
                        "Sign": [significant_sign],
                        "Skill": [
                            (
                                1
                                - df_res_merged.nRMSEReference
                                / df_res_merged.nRMSEBaseline
                            ).mean()
                        ],
                        "StdSkill": [
                            (
                                1
                                - df_res_merged.nRMSEReference
                                / df_res_merged.nRMSEBaseline
                            ).std()
                        ],
                    }
                )
                df_res = add_naming_columns_inductive_tl(df_res, rel_files[idx].stem)
                all_results += df_res
    return all_results


def df_to_latex_file(df, file_name, caption=""):
    if caption == "":
        caption = "TODO"
    prefix = (
        "\\begin{table}\n"
        + "\\caption{"
        + caption
        + "}\n"
        + "\\begin{center}\n"
        + "\\resizebox{\\textwidth}{!}{%\n"
    )
    postfix = "\n}\n\\end{center}\n \\end{table}"
    with open(file_name, "w") as text_file:
        text_file.write(prefix + df_to_latex(df)[:-1] + postfix)