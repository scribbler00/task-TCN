__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader


from fastai.torch_core import *
from fastai.tabular import *
from fastai.tabular.core import *

from fastai.tabular.core import (
    TabularPandas,
    Categorify,
    Normalize,
    FillMissing,
    IndexSplitter,
    MaskSplitter,
    RandomSplitter,
    range_of,
)
from fastai.tabular.all import *

from .utils_pytorch import np_to_dev
from .utils import np_int_dtypes
from fastcore.basics import listify


__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


def get_samples_per_day(df):
    """
    Extract the amount of entries per day from the DataFrame
    Parameters
    ----------
    df : pandas.DataFrame
        the DataFrame used for the conversion.

    Returns
    -------
    integer
        amount of entries per day.
    """
    samples_per_day = -1
    mins = 0
    for i in range(1, 10):
        mins = (df.index[-i] - df.index[-(i + 1)]).seconds // 60
        # 15 min resolution
        if mins == 15:
            samples_per_day = 24 * 4
            break
        # hourly resolution
        elif mins == 60:
            samples_per_day = 24
            break
        # three hour resolution
        elif mins == 180:
            samples_per_day = 8
            break
    if samples_per_day == -1:
        raise ValueError(f"{mins} is an unknown sampling time.")

    return samples_per_day


def get_y_ranges(y: torch.Tensor):
    """
    Get the minimum and maximum values of the targets 'y'.

    Parameters
    ----------
    y : pytorch.Tensor/list/pandas.DataFrame
        target values.

    Returns
    -------
    list of tuples
        the x-th tuple contains the minimum and maximum value of 'y[x]'.
    """
    y = np_to_dev(y)

    y_ranges = []
    for i in range(0, y.shape[1]):
        max_y = torch.max(y[:, i]) * 1.2
        min_y = torch.min(y[:, i]) * 1.2
        y_range = (min_y.item(), max_y.item())
        y_ranges.append(y_range)

    return y_ranges


def tp_from_df_from_dtypes(
    df: pd.DataFrame,
    y_columns: list,
    standardize_X: bool = False,
    valid_percent: float = None,
    val_idxs: list = None,
    do_split_by_n_weeks: bool = False,
    every_n_weeks: int = 4,
    for_n_weeks: int = 1,
    skip_first: bool = True,
    categorify=True,
    add_y_to_x=False,
    add_x_to_y=False,
):
    """
    Transform a pandas.DataFrame into a fastai TabularPandas

    The parameters are used to determine how to split the data into training/validation data, and which features to
    use are input/target.
    Parameters
    ----------
    df : pandas.DataFrame
        the DataFrame used for the conversion.
    y_columns : string or list of strings
        name of the columns which are used as targets.
    standardize_X : bool, default=False
        if True, the features of the DataSet are standardized.
    valid_percent : float
        split the data randomly, with 'valid_percent' denoting the amount of data used as validation data (in percent).
    val_idxs : list
        split the data according to the given indizes.
    do_split_by_n_weeks : bool
        determines whether the DataFrame is to be split into training- and validation data by a given frequency.
    every_n_weeks : integer
        determines the frequency at which entries are added (in weeks) as validation data.
    for_n_weeks : integer
        determines how many weeks are to be processed each 'every_n_weeks' as validation data.
    skip_first : bool
        determines whether the first week should be skipped as validation data, as it could be shorter than the rest (start of year).
    categorify : bool
        determines whether the data is to be categorized in the manner of pandas Categorical.
    add_y_to_x : bool
        determines weather y will be also in x or not (used for generating artifical data with GANs)

    Returns
    -------
    fastai TabularPandas
        A modified pandas.DataFrame, which contains information about which data is used for training/validation,
        and which features are used as input/target.
    """

    y_columns = listify(y_columns)
    # Replace missing values by the median of the column
    procs = [FillMissing]

    if categorify:
        procs += [Categorify]

    if standardize_X:
        procs += [Normalize]

    if not add_y_to_x:
        x_cat_columns = df.drop(y_columns, axis=1).columns
    else:
        x_cat_columns = df.columns

    dtypes = df[x_cat_columns].dtypes
    dtypes = dict(zip(x_cat_columns, dtypes))

    x_columns = []
    cat_columns = []

    for cur_col_name in dtypes.keys():
        cur_dtype = dtypes[cur_col_name]

        if cur_dtype in np_int_dtypes:
            cat_columns.append(cur_col_name)
        else:
            x_columns.append(cur_col_name)

    splits = None
    # Create a mask to split the data into training- and validation-data
    if do_split_by_n_weeks:
        _, idx_val = split_by_n_weeks(
            df,
            every_n_weeks=every_n_weeks,
            for_n_weeks=for_n_weeks,
            skip_first=skip_first,
        )
        splits = MaskSplitter(idx_val)(range_of(df))
    elif valid_percent is not None and valid_percent > 0.0:
        splits = RandomSplitter(valid_pct=valid_percent)(range_of(df))
    elif val_idxs is not None:
        splits = IndexSplitter(valid_idx=val_idxs)(range_of(df))

    if add_y_to_x:
        y_columns = x_columns

    if add_x_to_y:
        y_columns = x_columns

    to = TabularPandas(
        df,
        procs=procs,
        cat_names=cat_columns,
        cont_names=x_columns,
        y_names=y_columns,
        splits=splits,
        do_setup=True,
        inplace=True,
        y_block=RegressionBlock(),
    )

    return to


def tp_from_df(
    df: pd.DataFrame,
    y_columns: list,
    cat_columns: list = [],
    x_columns: list = [],
    valid_percent: float = None,
    do_split_by_n_weeks: bool = False,
    every_n_weeks: int = 4,
    for_n_weeks: int = 1,
    skip_first: bool = True,
    standardize_X: bool = False,
    categorify=True,
):
    """
    Transform a pandas.DataFrame into a fastai TabularPandas

    The parameters are used to determine how to split the data into training/validation data, and which features to
    use are input/target.
    Parameters
    ----------
    df : pandas.DataFrame
        the DataFrame used for the conversion.
    y_columns : string or list of strings
        name of the columns which are used as targets.
    cat_columns : list of strings
        names of the categorical features.
    x_columns : list of strings
        names of the non-categorical features. If empty, all features except the targets and categorical features are
        used.
    valid_percent : float
        split the data randomly, with 'valid_percent' denoting the amount of data used as validation data (in percent).
    do_split_by_n_weeks : bool
        determines whether the DataFrame is to be split into training- and validation data by a given frequency.
    every_n_weeks : integer
        determines the frequency at which entries are added (in weeks).
    for_n_weeks : integer
        determines how many weeks are to be processed each 'every_n_weeks'.
    skip_first : bool
        determines whether the first week should be skipped, as it could be shorter than the rest (start of year).
    standardize_X : bool, default=False
        if True, the features of the DataSet are standardized.
    categorify : bool
        determines whether the data is to be categorized in the manner of pandas Categorical.

    Returns
    -------
    fastai TabularPandas
        A modified pandas.DataFrame, which contains information about which data is used for training/validation,
        and which features are used as input/target.
    """

    x_columns = listify(x_columns)
    cat_columns = listify(cat_columns)
    y_columns = listify(y_columns)

    procs = [FillMissing]

    if categorify:
        procs += [Categorify]

    if standardize_X:
        procs += [Normalize]

    if len(x_columns) == 0:
        x_columns = [c for c in df.columns if c not in y_columns + cat_columns]

    splits = None
    # Create a mask to split the data into training- and validation-data
    if do_split_by_n_weeks:
        _, idx_val = split_by_n_weeks(
            df,
            every_n_weeks=every_n_weeks,
            for_n_weeks=for_n_weeks,
            skip_first=skip_first,
        )
        splits = MaskSplitter(idx_val)(range_of(df))
    elif valid_percent is not None and valid_percent > 0:
        splits = RandomSplitter(valid_pct=valid_percent)(range_of(df))

    to = TabularPandas(
        df,
        procs=procs,
        cat_names=cat_columns,
        cont_names=x_columns,
        y_names=y_columns,
        splits=splits,
        do_setup=True,
        inplace=True,
        y_block=RegressionBlock(),
    )

    return to


def split_by_year(df: pd.DataFrame, year: str = "2016"):
    """
    Split the DataFrame in two, at the first day of the given year.

    Parameters
    ----------
    df : pandas.DataFrame
        the DataFrame used for the conversion.
    year : string / integer
        the year at which the data is to be split

    Returns
    -------
    pandas.DataFrame
        The first part of the input DataFrame.
    pandas.DataFrame
        The second part of the input DataFrame.
    """
    year = str(year)
    mask = df.index < pd.to_datetime(f"{year}-01-01", utc=True)

    return df[mask], df[~mask]


def split_by_n_weeks(
    cur_dataset: pd.DataFrame,
    every_n_weeks: int = 4,
    for_n_weeks: int = 1,
    skip_first: bool = True,
):
    """
    Create an array containing bools for each entry in cur_dataset, with alternating batches of True- and False-values.

    Parameters
    ----------
    cur_dataset : pandas.DataFrame
        the DataFrame used for the conversion.
    every_n_weeks : integer
        determines the frequency at which entries are added (in weeks).
    for_n_weeks : integer
        determines how many weeks are to be processed each 'every_n_weeks'.
    skip_first : bool
        determines whether the first week should be skipped, as it could be shorter than the rest (start of year).

    Returns
    -------
    numpy.array
        inverted mask.
    numpy.array
        Array containing a bool for each entry in 'cur_dataset', denoting whether it is in the range defined
        by 'every_n_weeks', 'for_n_weeks'.
    """
    # Works for pd.Dataframes and Datasets, every_n_weeks and for_n_weeks can also be floats

    start_date = np.min(cur_dataset.index)
    end_date = np.max(cur_dataset.index)

    sw = pd.date_range(
        start_date, end_date, freq=f"{int(every_n_weeks*7)}D", normalize=True
    )
    # ew is the same as sw with an offset determined by for_n_weeks
    ew = pd.date_range(
        start_date + pd.DateOffset(days=int(for_n_weeks * 7)),
        end_date,
        freq=f"{(every_n_weeks)*7}D",
        normalize=True,
    )

    # In case the first elements in the DataFrame are not a full week (just the last few days)
    # the week's length in the DataFrame is inconsistent with the other weeks, and should be removed
    if skip_first:
        sw = sw[1:]
        ew = ew[1:]

    mask = np.zeros(len(cur_dataset.index), dtype=np.bool)

    for s, e in zip(sw, ew):
        mask = mask | ((cur_dataset.index > s) & (cur_dataset.index < e))

    indices = np.arange(len(cur_dataset))

    return ~mask, mask


def create_consistent_number_of_sampler_per_day(
    df: pd.DataFrame, num_samples_per_day: int = 24
) -> pd.DataFrame:
    """
    Remove days with less than the specified amount of samples from the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        the DataFrame used for the conversion.
    num_samples_per_day : integer
        the amount of samples each day in the DataFrame.

    Returns
    -------
    pandas.DataFrame
        the given DataFrame, now with a consistent amount of samples each day.
    """

    # Create a list of booleans, where each day with 'less than num_samples_per_day' samples is denoted with 'True'
    mask = df.resample("D").apply(len).PowerGeneration
    mask = (mask < num_samples_per_day) & (mask > 0)

    for i in range(len(mask)):
        if mask[i]:
            new_day = mask.index[i] + pd.DateOffset(days=1)
            new_day.hours = 0

            cur_mask = (df.index < mask.index[i]) | (df.index >= new_day)
            df = df[cur_mask]

    mask = df.resample("D").apply(len).PowerGeneration
    mask = (mask < num_samples_per_day) & (mask > 0)

    if mask.sum() != 0:
        raise ValueError("Wrong sample frequency.")

    return df


class Timeseries(fastuple):
    def show(self, ctx=None, **kwargs):
        # TODO: make subplots with grid
        n_samples = 5

        (
            cat,
            x,
            y,
        ) = self
        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
            y = y.reshape(1, y.shape[0], y.shape[1])

        #  for every feature
        for idx in range(x.shape[1]):
            # and n samples
            for idy in range(min(n_samples, x.shape[0])):
                plt.plot(x[idx, idy, :], alpha=0.2)
            plt.title(f"Feature {idx} Samples")
            plt.show()

        #  for every target
        for idx in range(y.shape[1]):
            # and n samples
            for idy in range(min(n_samples, y.shape[0])):
                plt.plot(y[idx, idy, :], alpha=0.2)
            plt.title(f"Target {idx} Samples")
            plt.show()


class TimeseriesTransform(Transform):
    def __init__(
        self,
        tp: TabularPandas,
        timeseries_length: int = 24,
        batch_first: bool = True,
        sequence_last: bool = True,
        drop_last: bool = True,
        is_train: bool = False,
        is_valid: bool = False,
        drop_inconsistent_cats: bool = True,
        y_timeseries_offset=0,
        step_size=24,
    ):
        """

        Parameters
        ----------
        tp : fastai TabularPandas
            input dataset, has to be set up before.
        timeseries_length : integer
            amount of elements in each time series used for the neural network.
        batch_first : bool
            determines whether the first dimension of the resulting Tensors denoted the batch.
        sequence_last : bool
            determines whether the last dimension of the resulting Tensors denoted the sequence length.
        drop_last : bool
            determines whether the last elements of the TabularPandas should be dropped, if they are not enough
            to form another time series.
        is_train : bool
            determines whether the current instance is used for training. Mutually exclusive with 'is_valid'.
        is_valid : bool
            determines whether the current instance is used for validation. Mutually exclusive with 'is_train'.
        drop_inconsistent_cats : bool
        """
        if is_train and is_valid:
            raise ValueError("Cannot be train and validation data at the same time.")

        if not is_train and not is_valid:
            self.tp = tp
        elif is_train:
            self.tp = tp.train
        elif is_valid:
            self.tp = tp.valid

        timeseries_length = listify(timeseries_length)
        if len(timeseries_length) == 1:
            # first is considere for x/cat and second one for y
            timeseries_length = L(timeseries_length[0], timeseries_length[0])

        self.timeseries_length = timeseries_length
        self.step_size = step_size
        self.batch_first = batch_first
        self.sequence_last = sequence_last
        self.drop_last = drop_last
        self.has_cat = len(tp.cat_names) > 0
        self.drop_inconsistent_cats = drop_inconsistent_cats

        # Drop the last few elements if len('tp') % 'timeseries_length' != 0
        self._match_timeseries_length(drop_last)

        self._convert_tabular_pandas_to_timeseries(
            timeseries_length, y_timeseries_offset
        )
        self._check_categoricals()

        self._adjust_to_required_timeseries_representation()

    def _adjust_to_required_timeseries_representation(self):
        self.xs = TimeseriesTransform._adjust_ts_and_batch(
            self.xs, self.batch_first, self.sequence_last
        )
        self.ys = TimeseriesTransform._adjust_ts_and_batch(
            self.ys, self.batch_first, self.sequence_last
        )
        if self.has_cat:
            self.cats = TimeseriesTransform._adjust_ts_and_batch(
                self.cats, self.batch_first, self.sequence_last
            )

    def _convert_tabular_pandas_to_timeseries(
        self, timeseries_length, y_timeseries_offset
    ):
        len_data = len(self.tp.xs)

        ts_x_length = timeseries_length[0]
        ts_y_length = timeseries_length[1]

        max_samples = len_data - y_timeseries_offset - ts_y_length

        samples_tensor = max_samples // self.step_size + 1

        xs = np.zeros((samples_tensor, self.tp.conts.shape[1], ts_x_length))
        ys = np.zeros((samples_tensor, self.tp.ys.shape[1], ts_y_length))

        cats = None
        if self.has_cat:
            cats = np.zeros((samples_tensor, self.tp.cats.shape[1], ts_x_length))

        for sample_id, i in enumerate(range(0, max_samples + 1, self.step_size)):
            start_x = i
            end_x = start_x + ts_x_length
            start_y = i + y_timeseries_offset
            end_y = start_y + ts_y_length

            xs[sample_id, :, :] = (
                self.tp.conts[start_x:end_x].values.transpose().reshape(-1, ts_x_length)
            )
            if self.has_cat:
                cats[sample_id, :, :] = (
                    self.tp.cats[start_x:end_x]
                    .values.transpose()
                    .reshape(-1, ts_x_length)
                )

            ys[sample_id, :, :] = (
                self.tp.ys.iloc[start_y:end_y]
                .values.transpose()
                .reshape(-1, ts_y_length)
            )

        self.xs = tensor(xs)
        self.ys = tensor(ys)
        if cats is not None:
            self.cats = tensor(cats).long()
        else:
            self.cats = cats

    def _match_timeseries_length(self, drop_last):
        # Drop the last few elements if len('tp') % 'timeseries_length' != 0
        if drop_last:
            dividable_length = range(
                0,
                self.tp.xs.shape[0]
                // self.timeseries_length[0]
                * self.timeseries_length[0],
            )
            self.tp = self.tp.iloc[dividable_length]

    def __len__(self):
        """
        Return the length of the used TabularPandas.

        Returns
        -------
        integer
            amount of lines in the TabularPandas
        """
        return self.xs.shape[0]

    def _check_categoricals(self):
        # If selected, drop all categorical columns which do not have a constant value for each time series
        # individually.
        if self.drop_inconsistent_cats and self.has_cat:
            if self.batch_first and self.sequence_last:
                for i in range(self.cats.shape[1]):
                    keep_indexes = []
                    for j in range(self.cats.shape[0]):
                        if (self.cats[j, i, :] - self.cats[j, i, 0]).sum() == 0:
                            keep_indexes += [j]

                    n_dropped = self.cats.shape[0] - len(keep_indexes)
                    if n_dropped > 0:
                        warnings.warn(
                            f"Dropped {n_dropped} elements due to inconsistent categoricals in a sequence."
                        )
                    self.xs = self.xs[keep_indexes, :, :]
                    self.cats = self.cats[keep_indexes, :, :]
                    self.ys = self.ys[keep_indexes, :, :]
            elif self.has_cat:
                raise NotImplementedError(
                    f"Drop inconsistent categoricals is not implemented for batch_first {self.batch_first} and sequence_last {self.sequence_last}"
                )

    @property
    def no_input_features(self):
        """
        Return the amount of continuous features in the TabularPandas.
        Returns
        -------
        integer
            amount of continuous features in the TabularPandas
        """
        return len(self.tp.cont_names)

    @staticmethod
    def _adjust_ts_and_batch(data, batch_first, sequence_last):
        """
        Swap the dimensions of the given Tensor.
        Parameters
        ----------
        data : pytorch.Tensor
            Three dimensional Tensor whose dimensions are to be swapped. Expectes data of the dimension (batch, features, sequence length).
        batch_first : bool
            determines whether the first dimension of the resulting Tensors should denote the batch.
        sequence_last : bool
            determines whether the last dimension of the resulting Tensors should denote the sequence length.

        Returns
        -------
        data : pytorch.Tensor
            input tensor with swapped dimensions.
        """

        if batch_first and sequence_last:
            # batch, feature, seq -> batch, seq, feature
            pass
        elif batch_first and not sequence_last:
            # batch, feature, seq -> batch, seq, feature
            data = data.permute(0, 2, 1)
        elif not batch_first and not sequence_last:
            # batch, feature, seq -> seq, batch, feature
            data = data.permute(2, 0, 1)

        return data

    @staticmethod
    def _convert_to_batch_ts_feature_view(data, batch_first, sequence_last):
        """
        Converts the data to the followong dimension (batch, sequence length, features).

        Parameters
        ----------
        data : pytorch.Tensor
            three dimensional Tensor whose dimensions are to be swapped.
        batch_first : bool
            determines whether the first dimension of the resulting Tensors denotes the batch.
        sequence_last : bool
            determines whether the last dimension of the resulting Tensors denotes the sequence length.

        Returns
        -------
        data : pytorch.Tensor
            input tensor with dimensions to be swapped.
        """

        if batch_first and sequence_last:
            # batch, feature, seq -> batch, seq, feature
            data = data.permute(0, 2, 1)
        elif not batch_first and not sequence_last:
            # seq, batch, feature -> batch, seq, feature
            data = data.permute(1, 0, 2)
        elif not batch_first and sequence_last:
            # feature, batch, seq -> batch, seq, feature
            data = data.permute(1, 2, 0)

        return data

    def show(self, max_n=10, **kwargs):
        """
        Create a plot for a 'max_n' of input- and target time series.
        Parameters
        ----------
        max_n : interger
            amount of samples that are to be plotted.
        kwargs

        Returns
        -------

        """
        # TODO: make subplots with grid
        tmp_data = TimeseriesTransform._adjust_ts_and_batch(
            self.xs, batch_first=True, sequence_last=True
        )

        for idx in range(self.xs.shape[1]):
            plt.plot(self.xs[0:max_n, idx, :].reshape(-1, 1))
            plt.title(f"Feature {self.tp.cont_names[idx]}")
            plt.show()

        for idx in range(self.ys.shape[1]):
            plt.plot(self.ys[0:max_n, idx, :].reshape(-1, 1))
            plt.title(f"Target {self.tp.y_names[idx]}")
            plt.show()

    def to_tfmd_lists(self, splits=None):
        """
        Transform the instance into a fastai TfmdDL
        Parameters
        ----------
        splits : fastai.data.transforms
            a function to define how to split the indizes of the data into training- and validation-data.
        Returns
        -------
        fastai TfmdDL
            transformed instance.
        """
        if splits is None:
            return TfmdLists(
                range(len(self)),
                self,
            )
        else:
            return TfmdLists(range(len(self)), self, splits=splits(range(len(self))))

    def to_dataloaders(
        self, bs=32, splits=None, shuffle=True, drop_last=False, device="cpu"
    ):
        """
        Transform the instance into a fastai DataLoader
        Parameters
        ----------
        bs : integer
            defines the batch size for the DataLoader.
        splits : fastai.data.transforms
            a function to define how to split the indizes of the data into training- and validation-data.
        shuffle : bool
            determines whether the data is to be shuffled each time the DataLoader is fully read/iterated.
        drop_last : bool
            determines whether the last elements of the TabularPandas should be dropped, if they are not enough
            to form another time series.
        device : pytorch.device
            determines which device the calculations are to be performed on.

        Returns
        -------
        fastai DataLoader
            the DataLoader for the current instance, allowing for handling data and training in combination
            with pytorch.
        """
        if splits is not None:
            data = self.to_tfmd_lists(splits)
            dl = DataLoaders.from_dsets(
                data.train,
                data.valid,
                bs=bs,
                shuffle=shuffle,
                drop_last=drop_last,
                device=device,
            )
        else:
            data = self.to_tfmd_lists()
            dl = DataLoaders.from_dsets(
                data, None, bs=bs, shuffle=shuffle, drop_last=drop_last, device=device
            )

        return dl

    def encodes(self, i: int):
        """

        Parameters
        ----------
        i : interger
            index of the time series to be processed.

        Returns
        -------
        Timeseries
            Timeseries object of the selected time series.
        """
        if self.has_cat:
            ts = Timeseries(
                # TODO: could be TensorCategory, but do we need to do that for each category?
                tensor(self.cats[i]).long(),
                tensor(self.xs[i]),
                tensor(self.ys[i]),
            )
        else:
            ts = Timeseries(tensor([]), self.xs[i], self.ys[i])

        ts.cont_names = self.tp.cont_names

        return ts

    def decode(self):
        """
        ToDo, not sure about this functionality yet
        Revert the transformations of the TabularPandas defined by procs.
        Returns
        -------
        TabularPandas
            input TabularPandas with reverted transformations
        """
        return self.tp.procs.decode(self.tp)

    def new(self, data):
        """
        Create a new instance of this class, with the current values as template.
        Parameters
        ----------
        data : pandas.DataFrame/fastai TabularPandas.

        Returns
        -------
        TimeseriesTransform
            TimeseriesTransform of the given data.
        """
        new_inst = None
        if isinstance(data, pd.DataFrame):
            new_inst = self.tp.new(data)
        elif isinstance(data, TabularPandas):
            new_inst = data
        else:
            raise TypeError

        new_inst.process()

        new_inst = TimeseriesTransform(
            new_inst,
            timeseries_length=self.timeseries_length,
            sequence_last=self.sequence_last,
            batch_first=self.batch_first,
            check_consistent_number_per_days=self.check_consistent_number_per_days,
            drop_last=self.drop_last,
            is_train=False,
            is_valid=False,
        )

        return new_inst

    def decodes(self, row):
        """

        Parameters
        ----------
        ToDo
        row : list


        Returns
        -------
        pandas.DataFrame
            DataFrame containing the data of rows, with the dimension swap being reverted.
        """
        cat = None

        cat, cont, ys = row

        cat = TimeseriesTransform._return_to_batch_ts_feature_view(
            cat, self.batch_first, self.sequence_last
        )
        cont = TimeseriesTransform._return_to_batch_ts_feature_view(
            cont, self.batch_first, self.sequence_last
        )

        ys = TimeseriesTransform._return_to_batch_ts_feature_view(
            ys, self.batch_first, self.sequence_last
        )

        cat = to_np(cat.reshape(-1, cat.shape[2]))
        cont = to_np(cont.reshape(-1, cont.shape[2]))
        ys = to_np(ys.reshape(-1, ys.shape[2]))

        print(cat.shape, cont.shape, ys.shape)

        df = pd.DataFrame(np.concatenate([cat, cont, ys], axis=1))
        df.columns = self.tp.cat_names + self.tp.cont_names + self.tp.y_names

        #  drop duplicated columns, e.g., in case of autoencoder
        df = df.loc[:, ~df.columns.duplicated()]

        return self.tp.new(df).decode()


@typedispatch
def show_batch(
    x: Timeseries,
    y,
    samples,
    ctxs=None,
    max_n=6,
    nrows=None,
    ncols=2,
    figsize=None,
    **kwargs,
):
    # TODO: create grid plots?
    #     if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    #     if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    #     for i,ctx in enumerate(ctxs): Timeseries(x[0][i], x[1][i], x[2][i]).show(ctx=ctx)
    x.show()


def create_processed_dl(
    tp: TabularPandas,
    df: pd.DataFrame,
    bs: int = 16,
    valid_percent=0.1,
    shuffle_train=True,
):
    """
    Create a new fastai DataLoader, using the given TabularPandas as a template.
    Parameters
    ----------
    tp : TabularPandas
        template for the new data.
    df : pandas.DataFrame
        new data that is to be transformed.
    bs : integer
        defines the batch size for the DataLoader.
    valid_percent
    shuffle_train : bool
            determines whether the data is to be shuffled each time the DataLoader is fully read/iterated.

    Returns
    -------
    fastai DataLoader
        the DataLoader for the given data and TabularPandas template, allowing for handling data and training in
        combination with pytorch.

    """
    tp_new = tp.new(df)
    tp_new.process()
    tp_new = tp_from_df_from_dtypes(
        tp_new.items,
        "PowerGeneration",
        valid_percent=valid_percent,
        standardize_X=False,
    )
    # dls = tp_new.dataloaders(bs=bs, pin_memory=True, shuffle_train=shuffle_train)
    dls = tp_new.dataloaders(bs=bs, pin_memory=True, shuffle=shuffle_train)

    return dls
