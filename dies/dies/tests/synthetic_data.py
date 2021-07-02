import numpy as np
import pandas as pd

from fastai.data.all import *


class SyntheticData:
    def __init__(
        self, n_samples, n_features, n_targets, val_perc, batch_size, add_dim=False
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_targets = n_targets
        self.val_perc = val_perc
        self.batch_size = batch_size
        self.add_dim = add_dim

    def get_df(self):
        data = np.random.normal(0, 1, (self.n_samples, self.n_features))
        x_columns = [f"x{idx}" for idx in range(self.n_features)]
        df = pd.DataFrame(data, columns=x_columns)
        y_columns = [f"y{idx}" for idx in range(self.n_targets)]
        for y in y_columns:
            df[y] = df[x_columns].sum(axis=1) + np.random.normal(
                0, 0.25, self.n_samples
            )
        return df

    def get_dls(self):
        blocks = (RegressionBlock, RegressionBlock)
        splitter = RandomSplitter(valid_pct=self.val_perc)

        def get_item(df):
            return df.values

        def get_x(d):
            x = d[: self.n_features]
            if self.add_dim:
                x = np.expand_dims(x, axis=1)
            return x

        def get_y(d):
            y = d[-self.n_targets :]
            if self.add_dim:
                y = np.expand_dims(y, axis=1)
            return y

        db = DataBlock(
            blocks=blocks,
            splitter=splitter,
            get_items=get_item,
            get_x=get_x,
            get_y=get_y,
        )
        dls = db.dataloaders(self.get_df(), bs=self.batch_size)
        return dls
