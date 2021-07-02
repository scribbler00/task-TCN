import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

class FilterWeather(BaseEstimator, TransformerMixin):
    """
        Module to add column with bins for features based on given limits for the bins.

        Parameters
        ----------

        column_names : List (by default ['WindSpeed10m'])
            List with column names that a binned feature should be created for.

        limits : List of lists(by default [[6, 11, 20]])
            List of lists containing the limits for each column/feature.

    """

    def __init__(self):
        filter_values = defaultdict(list)
        filter_values['ZonalWind122m'] = [-100,100]
        filter_values['MeridionalWind122m']=[-100,100]
        filter_values['ZonalWind73m']=[-100,100]
        filter_values['MeridionalWind73m']=[-100,100]
        filter_values['ZonalWind36m']=[-100,100]
        filter_values['MeridionalWind36m']=[-100,100]
        filter_values['Temperature122m']=[200,340]
        filter_values['Temperature36m']=[200,340]
        
        self.filter_values = filter_values


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
    
    def __weather_filter(self, df, column_name='MeridionalWind122m', lower_limit=-100, upper_limit=100):
        mask = (df.loc[:,column_name] > lower_limit) & (df.loc[:,column_name] < upper_limit)
        return mask.values


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

        #df_c = df_cosmo.copy()
        for key in self.filter_values.keys():

            mask = (self.__weather_filter(transformed_data, key, self.filter_values[key][0], self.filter_values[key][1]))
            #print(key, np.sum(~ mask))
            transformed_data = transformed_data[mask]
    
        return transformed_data
