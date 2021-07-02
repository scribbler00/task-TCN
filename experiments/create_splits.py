import numpy as np
import json
from sklearn.model_selection import KFold

from confer.utils import str_to_path, get_blacklist


server = True

if server:
    folders = [
        "/mnt/work/prophesy/prophesy-data/WindSandbox2015/",
        "/mnt/work/prophesy/prophesy-data/PVSandbox2015",
    ]
    base_folder = "/mnt/work/transfer/conv_transfer/"
else:
    folders = [
        "~/data/prophesy-data/WindSandbox2015/",
        "~/data/prophesy-data/PVSandbox2015/",
    ]
    base_folder = "./results/"

names = [
    "2015_wind",
    "2015_pv",
]


def create_splits(folder, res_folder):
    data_folder = str_to_path(folder)
    files = data_folder.ls()
    # files = [f for f in files if f.suffix == ".h5" or f.suffix == ".csv"]
    files_new = []
    blacklist = get_blacklist()
    for file in files:
        file_string = str(file)
        if file.stem in blacklist:
            print("Skipped:", file_string)
            continue

        if (str(file).endswith(".csv") or str(file).endswith(".h5")) and (
            (
                (
                    "solar" not in file_string.lower()
                    and "wind" in str(res_folder).lower()
                )
                or ("solar" in file_string.lower() and "pv" in str(res_folder).lower())
            )
            or "gefcom" not in str(folder).lower()
        ):
            files_new.append(file)
    files = files_new

    n_splits = 5
    if len(files) < n_splits:
        n_splits = len(files)
    kf = KFold(n_splits=n_splits, shuffle=True)
    splits = dict()
    for run_id, (source_file_ids, target_file_ids) in enumerate(kf.split(files)):

        splits[run_id] = [str(f) for f in np.array(files)[source_file_ids]]

    str_to_path(f"{base_folder}/{res_folder}")
    with open(f"{base_folder}/{res_folder}/splits.json", "w") as f:
        json.dump(splits, f)


for folder_input, folder_ouput in zip(folders, names):
    create_splits(folder_input, folder_ouput)
