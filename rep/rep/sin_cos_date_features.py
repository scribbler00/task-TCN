import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class SinCosFeatures(BaseEstimator, TransformerMixin):
    """
        Module to add column with sinus and cosnis for features.

        Parameters
        ----------

        column_names : List (by default ['Hour'])
            List with column names that a sinus and cosinus feature should be created for.


    """

    def __init__(self, column_names=['Hour']):

        self.column_names = column_names

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
        """ Adds time difference since model run.
            ----------
            X : pandas DataFrame
                The DataFrame holding the numerical weather prediction data.
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
        transformed_data = X

        for id_column, feature_name in enumerate(self.column_names):

            transformed_data[feature_name + 'Sin'] = np.sin(transformed_data.loc[:, feature_name])
            transformed_data[feature_name + 'Cos'] = np.cos(transformed_data.loc[:, feature_name])

        return transformed_data
