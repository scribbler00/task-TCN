import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
        Module to scale pandas columns between 0 and 1.

        Parameters
        ----------

        features : list (by default ['WindSpeed100m'])
            Features that should be scaled beyween 0 and 1.

    """

    def __init__(self, features=['WindSpeed100m'], 
                 scaler_type='MinMaxScaler', 
                 ignore_features=None):
        """
            Constructor.

            Parameters
            ----------
            features : List
                The list holding the name of features to be scaled. 
                In case it is None, features are determined with
                pandas.columns and the ignored features, see ignore_features.
            scalter_type: String
                The string holding the type of sklearn.preprocessing scaler.
                Alloweded parameters: ['MinMaxScaler', 'StandardScaler', 'None']
            ignore_features : List
                List of features that will be ignored in the scaling. 
                In case features and ignore_features are None, all features are included.
        """
            
        
        self.features = features
        self.allowed_scalings = ['MinMaxScaler', 'StandardScaler']
        self.scaler_type = scaler_type
        self.ignore_features = ignore_features


    def fit(self, X, y=None):
        """
            Fitting function. Does nothing besides checking
            the validity of the DataFrames.

            Parameters
            ----------
            X : pandas DataFrame
                The DataFrame holding the data to scale.
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

            # determine correct features based on all features, remove the once that should be ignored
        if self.features is None:
            tmp_features = list(X.columns)
            for cur_feature in self.ignore_features:
                if cur_feature in tmp_features: tmp_features.remove(cur_feature)
            self.features = tmp_features

        self.transformer = dict()
        if self.scaler_type is not None:
            for f in self.features:
                scaler = self.__get_scaler()

                if X[f].dtype == 'int64' or X[f].dtype == 'object':
                    self.transformer[f] = scaler.fit(X[f].astype('float64').values.reshape(-1, 1))
                else:
                    self.transformer[f] = scaler.fit(X[f].values.reshape(-1, 1))

        return self

    def transform(self, X, y=None):
        """ Scales all features between 0 and 1.
            ----------
            X : pandas DataFrame
                The DataFrame holding the data to scale.
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
        transformed_data = X.copy()

        if self.scaler_type is not None:
            for f in self.features:
                if transformed_data[f].dtype == 'int64' or transformed_data[f].dtype == 'object':
                    transformed_data[f] = self.transformer[f].transform(transformed_data[f].astype('float64').values.reshape(-1, 1))
                else:
                    transformed_data[f] = self.transformer[f].transform(transformed_data[f].values.reshape(-1, 1))
        return transformed_data

    def __get_scaler(self):
        if self.scaler_type == 'MinMaxScaler':
            return MinMaxScaler(copy=False, feature_range=(0, 1))
        elif self.scaler_type == 'StandardScaler':
            return StandardScaler(copy=False)
        elif self.scaler_type == 'None':
            return None
        else:
            raise Warning('Unknown scaling type. Allowed params:' + self.allowed_scalings)
