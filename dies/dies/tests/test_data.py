# from dies.data import (
#     Dataset,
#     DatasetCategoricalData,
#     TimeseriesDataset,
#     TimeseriesDatasetCategoricalData,
#     combine_datasets,
#     ds_from_df,
#     ds_from_df_from_dtypes,
#     scale_datasets,
# )
from dies import data
import numpy as np
import pandas as pd
import unittest
from numpy.testing import assert_almost_equal, assert_array_less, assert_array_equal
from sklearn.datasets import load_iris
import torch
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_2_dfs():
    X, y, _ = datasets.make_regression(
        n_samples=50,
        n_features=2,
        bias=1000,
        n_informative=2,
        noise=10,
        coef=True,
        random_state=42,
    )

    df1 = pd.DataFrame(
        data=np.concatenate([X, y.reshape(-1, 1)], axis=1),
        columns=["feat1", "feat2", "target"],
    )
    cats = np.random.randint(low=0, high=10, size=(df1.shape[0], 2))
    df1["cat_1"] = cats[:, 0]
    df1["cat_2"] = cats[:, 1]

    index1 = pd.date_range("2000-01-01", "2000-06-01", periods=df1.shape[0])
    index1 = pd.to_datetime(index1, utc=True)
    df1.index = index1

    X, y, _ = datasets.make_regression(
        n_samples=53,
        n_features=2,
        bias=800,
        n_informative=2,
        noise=8,
        coef=True,
        random_state=42,
    )

    df2 = pd.DataFrame(
        data=np.concatenate([X, y.reshape(-1, 1)], axis=1),
        columns=["feat1", "feat2", "target"],
    )
    cats = np.random.randint(low=0, high=10, size=(df2.shape[0], 2))
    df2["cat_1"] = cats[:, 0]
    df2["cat_2"] = cats[:, 1]

    index2 = pd.date_range("2000-06-01", "2000-12-01", periods=df2.shape[0])
    index2 = pd.to_datetime(index2, utc=True)
    df2.index = index2

    return df1, df2


# class TestDataset(unittest.TestCase):
#     def setUp(self):
#         data = load_iris()
#         self.df = pd.DataFrame(
#             data=np.c_[data["data"], data["target"]],
#             columns=data["feature_names"] + ["target"],
#         )
#         self.x_columns = self.df.drop("target", axis=1).columns
#         self.y_columns = "target"

#         X = self.df.drop("target", axis=1).values
#         y = self.df["target"].values

#         self.index = pd.date_range("2000-06-01", "2000-12-01", periods=X.shape[0])
#         self.index = pd.to_datetime(self.index, utc=True)

#         self.ds = Dataset(
#             x=X,
#             y=y,
#             x_columns=self.x_columns,
#             y_columns=self.y_columns,
#             index=self.index,
#             standardize_X=False,
#             device="cpu",
#         )

#     def test_correct_storing_of_dataset(self):

#         self.assertIsInstance(self.ds.x_columns, list)
#         self.assertIsInstance(self.ds.y_columns, list)

#         self.assertIsInstance(self.ds.x, torch.FloatTensor)
#         self.assertIsInstance(self.ds.y, torch.FloatTensor)

#         self.assertEqual(1, len(self.ds.y_columns))
#         self.assertEqual(4, len(self.ds.x_columns))

#     def test_to_df(self):

#         df = self.ds.to_df()

#         self.assertEqual(df.shape[0], self.df.shape[0])
#         self.assertEqual(df.shape[1], self.df.shape[1])

#         self.assertCountEqual(self.df.columns, df.columns)
#         self.assertCountEqual(self.index, df.index)


# class TestTimeseriesDataset(unittest.TestCase):
#     def setUp(self):
#         data = load_iris()
#         self.df = pd.DataFrame(
#             data=np.c_[data["data"], data["target"]],
#             columns=data["feature_names"] + ["target"],
#         )
#         self.x_columns = self.df.drop("target", axis=1).columns
#         self.y_columns = "target"

#         self.X = self.df.drop("target", axis=1).values
#         self.y = self.df["target"].values
        


#         self.x_shape = self.X.shape
        
#         self.y_shape = self.y.shape

#         self.index = pd.date_range("2000-06-01", "2000-12-01", periods=self.X.shape[0])
#         self.index = pd.to_datetime(self.index, utc=True)

        
#         self.timeseries_length=24
#         self.drop_last=True
#         self.batch_first = True
#         self.sequence_last = False

#         self.reduced_shape = self.df.shape[0]//self.timeseries_length*self.timeseries_length
#         self.reduced_index = self.index[range(self.reduced_shape)] if self.drop_last else self.index

#         self.ds = self._create_dataset(self.X, self.y, self.timeseries_length, self.batch_first, self.sequence_last)


#     def _create_dataset(self, X, y, ts_length, batch_first, sequence_last):
#         ds = TimeseriesDataset(
#             x=X,
#             y=y,
#             x_columns=self.x_columns,
#             y_columns=self.y_columns,
#             index=self.index,
#             standardize_X=False,
#             drop_last = self.drop_last,
#             timeseries_length = ts_length,
#             batch_first=batch_first,
#             sequence_last=sequence_last,
#             device="cpu",
#         )
#         return ds

#     def test_correct_storing_of_dataset(self):

#         self.assertIsInstance(self.ds.x_columns, list)
#         self.assertIsInstance(self.ds.y_columns, list)

#         self.assertIsInstance(self.ds.x, torch.FloatTensor)
#         self.assertIsInstance(self.ds.y, torch.FloatTensor)

#         self.assertEqual(1, len(self.ds.y_columns))
#         self.assertEqual(4, len(self.ds.x_columns))

#     def test_to_df(self):

#         df = self.ds.to_df()

#         self.assertEqual(df.shape[0], self.reduced_shape if self.drop_last else self.df.shape[0])

#         self.assertEqual(df.shape[1], self.df.shape[1] )

#         self.assertCountEqual(self.df.columns, df.columns)
#         index_to_use = self.reduced_index if self.drop_last else self.index 
#         self.assertCountEqual(index_to_use, df.index)
    
#     def test_correct_shapes(self):
#         # test standard dataset (batch_first=True, sequence_last=False)
#         self.assertEqual(24,self.ds.x.shape[1])
#         self.assertEqual(4,self.ds.x.shape[2])

#         ds =  self._create_dataset(self.X,self.y,self.timeseries_length, batch_first=False,sequence_last=True)
#         self.assertEqual(24,ds.x.shape[2])
#         self.assertEqual(4,ds.x.shape[0])

#         ds =  self._create_dataset(self.X,self.y,self.timeseries_length, batch_first=False,sequence_last=False)
#         self.assertEqual(24,ds.x.shape[0])
#         self.assertEqual(4,ds.x.shape[2])

#         ds =  self._create_dataset(self.X,self.y,self.timeseries_length, batch_first=True,sequence_last=True)
#         self.assertEqual(24,ds.x.shape[2])
#         self.assertEqual(4,ds.x.shape[1])

#         # Test out different time series lengths
#         ds =  self._create_dataset(self.X,self.y,ts_length=5, batch_first=True,sequence_last=False)
#         self.assertEqual(5,ds.x.shape[1])
#         self.assertEqual(4,ds.x.shape[2])
        
#         ds =  self._create_dataset(self.X,self.y,ts_length=30, batch_first=True,sequence_last=False)
#         self.assertEqual(30,ds.x.shape[1])
#         self.assertEqual(4,ds.x.shape[2])



# class TestDatasetCategorical(unittest.TestCase):
#     def setUp(self):
#         data = load_iris()
#         self.df = pd.DataFrame(
#             data=np.c_[data["data"], data["target"]],
#             columns=data["feature_names"] + ["target"],
#         )
#         cats = np.random.randint(
#             low=0, high=10, size=(self.df.shape[0], 2), dtype=np.integer
#         )
#         self.df["cat_1"] = cats[:, 0]
#         self.df["cat_2"] = cats[:, 1]

#         self.y_columns = "target"
#         self.cat_columns = ["cat_1", "cat_2"]
#         self.x_columns = self.df.drop(
#             [self.y_columns] + self.cat_columns, axis=1
#         ).columns

#         X = self.df[self.x_columns].values
#         cats = self.df[self.cat_columns].values
#         y = self.df[self.y_columns].values

#         self.index = pd.date_range("2000-06-01", "2000-12-01", periods=X.shape[0])
#         self.index = pd.to_datetime(self.index, utc=True)

#         self.ds = DatasetCategoricalData(
#             x=X,
#             cat=cats,
#             y=y,
#             index=self.index,
#             x_columns=self.x_columns,
#             cat_columns=self.cat_columns,
#             y_columns=self.y_columns,
#             standardize_X=False,
#             device="cpu",
#         )

#     def test_correct_storing_of_dataset(self):

#         self.assertIsInstance(self.ds.x_columns, list)
#         self.assertIsInstance(self.ds.y_columns, list)

#         self.assertIsInstance(self.ds.x, torch.FloatTensor)
#         self.assertIsInstance(self.ds.y, torch.FloatTensor)
#         self.assertIsInstance(self.ds.cat, torch.LongTensor)

#         self.assertEqual(1, len(self.ds.y_columns))
#         self.assertEqual(4, len(self.ds.x_columns))

#     def test_to_df(self):

#         df = self.ds.to_df()

#         self.assertEqual(df.shape[0], self.df.shape[0])

#         self.assertEqual(df.shape[1], self.df.shape[1])

#         self.assertCountEqual(self.df.columns, df.columns)
#         self.assertCountEqual(self.index, df.index)


# class TestDatasetCategoricalTimeseries(unittest.TestCase):
#     def setUp(self):
#         data = load_iris()
#         self.df = pd.DataFrame(
#             data=np.c_[data["data"], data["target"]],
#             columns=data["feature_names"] + ["target"],
#         )
#         cats = np.random.randint(
#             low=0, high=10, size=(self.df.shape[0], 2), dtype=np.integer
#         )
#         self.df["cat_1"] = cats[:, 0]
#         self.df["cat_2"] = cats[:, 1]

#         self.y_columns = "target"
#         self.cat_columns = ["cat_1", "cat_2"]
#         self.x_columns = self.df.drop(
#             [self.y_columns] + self.cat_columns, axis=1
#         ).columns

#         self.X = self.df[self.x_columns].values
#         self.cats = self.df[self.cat_columns].values
#         self.y = self.df[self.y_columns].values


#         self.x_shape = self.X.shape
#         self.cat_shape = self.cats.shape
#         self.y_shape = self.y.shape


#         self.index = pd.date_range("2000-06-01", "2000-12-01", periods=self.X.shape[0])
#         self.index = pd.to_datetime(self.index, utc=True)
#         self.timeseries_length=24
#         self.drop_last=True
#         self.batch_first = True
#         self.sequence_last = False

        
#         self.reduced_shape = self.df.shape[0]//self.timeseries_length*self.timeseries_length
#         self.reduced_index = self.index[range(self.reduced_shape)] if self.drop_last else self.index

#         self.ds = self._create_dataset(self.X,self.cats,self.y,self.timeseries_length,self.batch_first,self.sequence_last)

        
#     def _create_dataset(self, X, cats, y, ts_length, batch_first, sequence_last):
#         ds = TimeseriesDatasetCategoricalData(
#             x=X,
#             cat=cats,
#             y=y,
#             index=self.index,
#             x_columns=self.x_columns,
#             cat_columns=self.cat_columns,
#             y_columns=self.y_columns,
#             timeseries_length = ts_length,
#             drop_last=self.drop_last,
#             standardize_X=False,
#             sequence_last=sequence_last,
#             batch_first=batch_first,
#             device="cpu",
#         )
#         return ds

#     def test_correct_storing_of_dataset(self):

#         self.assertIsInstance(self.ds.x_columns, list)
#         self.assertIsInstance(self.ds.y_columns, list)

#         self.assertIsInstance(self.ds.x, torch.FloatTensor)
#         self.assertIsInstance(self.ds.y, torch.FloatTensor)
#         self.assertIsInstance(self.ds.cat, torch.LongTensor)

#         self.assertEqual(1, len(self.ds.y_columns))
#         self.assertEqual(4, len(self.ds.x_columns))
        
#     def test_correct_shapes(self):
#         # test standard dataset (batch_first=True, sequence_last=False)
#         self.assertEqual(24,self.ds.x.shape[1])
#         self.assertEqual(4,self.ds.x.shape[2])

#         ds =  self._create_dataset(self.X,self.cats,self.y,self.timeseries_length, batch_first=False,sequence_last=True)
#         self.assertEqual(24,ds.x.shape[2])
#         self.assertEqual(4,ds.x.shape[0])

#         ds =  self._create_dataset(self.X,self.cats,self.y,self.timeseries_length, batch_first=False,sequence_last=False)
#         self.assertEqual(24,ds.x.shape[0])
#         self.assertEqual(4,ds.x.shape[2])

#         ds =  self._create_dataset(self.X,self.cats,self.y,self.timeseries_length, batch_first=True,sequence_last=True)
#         self.assertEqual(24,ds.x.shape[2])
#         self.assertEqual(4,ds.x.shape[1])

#         # Test out different time series lengths
#         ds =  self._create_dataset(self.X,self.cats,self.y,ts_length=5, batch_first=True,sequence_last=False)
#         self.assertEqual(5,ds.x.shape[1])
#         self.assertEqual(4,ds.x.shape[2])
        
#         ds =  self._create_dataset(self.X,self.cats,self.y,ts_length=30, batch_first=True,sequence_last=False)
#         self.assertEqual(30,ds.x.shape[1])
#         self.assertEqual(4,ds.x.shape[2])

#     def test_to_df(self):

#         df = self.ds.to_df()

        
#         self.assertEqual(df.shape[0], self.reduced_shape if self.drop_last else self.df.shape[0])

#         self.assertEqual(df.shape[1], self.df.shape[1] )

#         self.assertCountEqual(self.df.columns, df.columns)
#         index_to_use = self.reduced_index if self.drop_last else self.index 
#         self.assertCountEqual(index_to_use, df.index)



# class TestDatasetGetYRanges(unittest.TestCase):
#     def setUp(self):
#         data = load_iris()
#         self.df = pd.DataFrame(
#             data=np.c_[data["data"], data["target"]],
#             columns=data["feature_names"] + ["target"],
#         )
#         self.df["target_2"] = self.df["target"] * 1.2

#         self.y_columns = ["target", "target_2"]
#         self.x_columns = self.df.drop(self.y_columns, axis=1).columns

#         X = self.df.drop(self.y_columns, axis=1).values
#         y = self.df[self.y_columns].values

#         self.ds = Dataset(
#             x=X,
#             y=y,
#             x_columns=self.x_columns,
#             y_columns=self.y_columns,
#             standardize_X=False,
#             device="cpu",
#         )

#     def test_get_y_ranges(self):
#         y_ranges = self.ds.y_ranges

#         self.assertEqual(y_ranges[0][0], 0)
#         self.assertEqual(y_ranges[0][1], 2.4)
#         self.assertEqual(y_ranges[1][0], 0)
#         self.assertEqual(y_ranges[1][1], 2.88)


# class TestCombineDataset(unittest.TestCase):
#     def setUp(self):
#         self.df1, self.df2 = get_2_dfs()

#         self.ds1 = ds_from_df(self.df1, "target")
#         self.ds2 = ds_from_df(self.df2, "target")

#     def test_combine_data(self):
#         ds_merged = combine_datasets([self.ds2, self.ds1])
#         df_merged = ds_merged.to_df()

#         self.assertEqual(self.df1.shape[0] + self.df2.shape[0], df_merged.shape[0])

#         for cur_index in self.df1.index:
#             self.assertIn(cur_index, df_merged.index)

#         for cur_index in self.df2.index:
#             self.assertIn(cur_index, df_merged.index)


# class TestScalingDataset(unittest.TestCase):
#     def setUp(self):
#         self.df1, self.df2 = get_2_dfs()

#         self.ds1 = ds_from_df(self.df1, "target")
#         self.ds2 = ds_from_df(self.df2, "target")

#     def test_inverse_scaling_with_new_scaling(self):
#         self.ds2.scale_input(scaler=StandardScaler)

#         x_sample = self.ds2.x[:, 0]
#         min_x = x_sample.min()
#         max_x = x_sample.max()

#         self.assertLessEqual(min_x, -1.1)
#         self.assertGreaterEqual(
#             max_x, 1.1,
#         )

#         self.ds2.scale_input(scaler=MinMaxScaler)

#         x_sample = self.ds2.x[:, 0]
#         min_x = x_sample.min()
#         max_x = x_sample.max()

#         self.assertGreaterEqual(min_x, -1e-9)
#         self.assertLessEqual(
#             max_x, 1.1,
#         )


# class TestMisc(unittest.TestCase):
#     def setUp(self):
#         self.df1, self.df2 = get_2_dfs()

#     def test_ds_from_df_from_dtypes(self):
#         ds = ds_from_df_from_dtypes(self.df1, "target")

#         self.assertCountEqual(["feat1", "feat2"], ds.x_columns)
#         self.assertCountEqual(["cat_1", "cat_2"], ds.cat_columns)
#         self.assertCountEqual(["target"], ds.y_columns)

#     def test_split_dataset(self):
#         ds = ds_from_df_from_dtypes(self.df1, "target")

#         ds_tr, ds_val, _ = data.train_test_split_dataset(cur_dataset=ds)

#         self.assertEqual(ds_tr.x.shape[0], 40)
#         self.assertEqual(ds_tr.x.shape[1], 2)
#         self.assertEqual(ds_tr.cat.shape[0], 40)
#         self.assertEqual(ds_tr.cat.shape[1], 2)
#         self.assertEqual(ds_tr.y.shape[0], 40)
#         self.assertEqual(ds_tr.y.shape[1], 1)
#         self.assertEqual(len(ds_tr.index), 40)

#         self.assertEqual(ds_val.x.shape[0], 10)
#         self.assertEqual(ds_val.x.shape[1], 2)
#         self.assertEqual(ds_val.cat.shape[0], 10)
#         self.assertEqual(ds_val.cat.shape[1], 2)
#         self.assertEqual(ds_val.y.shape[0], 10)
#         self.assertEqual(ds_val.y.shape[1], 1)
#         self.assertEqual(len(ds_val.index), 10)

#     def test_split_dataset_with_date(self):
#         ds = ds_from_df_from_dtypes(self.df1, "target")

#         ds_tr, ds_val, ds_te = data.train_test_split_dataset_with_test_date(
#             cur_dataset=ds, test_date="2000-03-01"
#         )

#         for index in ds_te.index:
#             self.assertGreaterEqual(index, pd.to_datetime("2000-03-01", utc=True))

#         for index in ds_tr.index:
#             self.assertLess(index, pd.to_datetime("2000-03-01", utc=True))

#         for index in ds_val.index:
#             self.assertLess(index, pd.to_datetime("2000-03-01", utc=True))

#         for index in ds_tr.index:
#             self.assertNotIn(index, ds_val.index)

#         for index in ds_val.index:
#             self.assertNotIn(index, ds_tr.index)

#     def test_scale_datasets(self):
#         ds1 = ds_from_df_from_dtypes(self.df1, "target")
#         ds2 = ds_from_df_from_dtypes(self.df2, "target")

#         x_sample = ds1.x[:, 0]
#         min_x = x_sample.min()
#         max_x = x_sample.max()

#         self.assertLessEqual(min_x, -1.1)
#         self.assertGreaterEqual(
#             max_x, 1.1,
#         )

#         x_sample = ds2.x[:, 0]
#         min_x = x_sample.min()
#         max_x = x_sample.max()

#         self.assertLessEqual(min_x, -1.1)
#         self.assertGreaterEqual(
#             max_x, 1.1,
#         )
