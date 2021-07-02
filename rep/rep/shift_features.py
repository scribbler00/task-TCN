import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# from pies.timeseries import SlidingWindow


class ShiftFeatures(BaseEstimator, TransformerMixin):
    """
    Module to add sliding window features.

    Parameters
    ----------

    sliding_window_features : List
        List with column names that shift features are created for.

    eval_position : String (default: 'past')
        String depicting  if past or future values should be shifted.


    """

    def __init__(
        self, sliding_window_features, eval_position="past", step_sizes=1, drop_na=True
    ):
        if not type(sliding_window_features) == list:
            sliding_window_features = [sliding_window_features]

        self.sliding_window_features = sliding_window_features
        if eval_position == "both":
            eval_position = ["future", "past"]
        if not isinstance(eval_position, list):
            eval_position = [eval_position]

        self.eval_position = eval_position
        if not isinstance(step_sizes, list):
            step_sizes = [step_sizes]
        self.step_sizes = step_sizes

        self.drop_na = drop_na

    def fit(self, X, y=None):
        """
        Fitting function. Does nothing besides checking
        the validity of the DataFrames.

        Parameters
        ----------
        X : pandas DataFrame
            The DataFrame holding the data.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        # check if X is pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data is no valid pandas DataFrame")
        return self

    def transform(self, X, y=None):
        """Adds time difference since model run.
        ----------
        X : pandas DataFrame
            The DataFrame holding the numerical weather prediction data and power generation.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """

        # check if X is pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data is no valid pandas DataFrame")

        # initialize transformed data.
        df_transformed = X

        for feature in self.sliding_window_features:
            for step_size in self.step_sizes:
                for shift_direction in self.eval_position:
                    tmp = np.ones(len(df_transformed[feature])) * np.nan

                    if shift_direction == "past":
                        tmp[step_size:] = df_transformed.loc[:, [feature]][
                            :-step_size
                        ].values.T
                        df_transformed[feature + "_t_minus_" + str(step_size)] = tmp

                    if shift_direction == "future":
                        tmp[0:-step_size] = df_transformed.loc[:, [feature]][
                            step_size:
                        ].values.T
                        df_transformed[feature + "_t_plus_" + str(step_size)] = tmp

        if self.drop_na:
            df_transformed = df_transformed.dropna(axis=0)

        return df_transformed

    def get_sliding_window_df(self, eval_pos, df_transformed, drop_na):
        sw = SlidingWindow(
            window_size=self.window_size,
            step_size=self.step_size,
            eval_position=eval_pos,
            timeseries_key_name="TimeUTC",
            dropna=drop_na,
        )

        sliding_window_df = sw.transform(
            df_transformed.loc[:, self.sliding_window_features]
        )

        return sliding_window_df
