from dies.utils import listify
import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


def measure_similarity(df1, df2, measure, columns, **kwargs):
    columns = listify(columns)

    results = []

    for cur_column in columns:
        results.append(_similarity(df1, df2, measure, cur_column, **kwargs))

    return results


def _similarity(df1, df2, measure, column, **kwargs):
    data1 = df1[column].values.ravel()
    data2 = df2[column].values.ravel()

    return measure(data1, data2, **kwargs)


def dtw(x_1, x_2):
    distance, path = fastdtw(x_1, x_2, dist=euclidean)
    return distance