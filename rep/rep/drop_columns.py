import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    """
        Module to remove specific columns.

    """

    def __init__(self, columns_to_drop=None):
        if columns_to_drop is not None and \
                type(columns_to_drop) is str:
            columns_to_drop = [columns_to_drop]

        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        """
            Fitting function. Does nothing besides checking
            the validity of the DataFrames.

            Parameters
            ----------
            X : pandas DataFrame
                The DataFrame holding the sensory data.
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
        """ Removes configured columns.
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

        if self.columns_to_drop is not None:
            transformed_data = X.drop([c for c in self.columns_to_drop if c in X.columns ], axis=1)
        else:
            transformed_data = X

        return transformed_data
