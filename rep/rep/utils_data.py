import copy
from fastai.data.transforms import RandomSplitter
from fastcore.basics import listify
from rep.utils_preprocessing import get_all_dfs, split_in_train_and_test

import pandas as pd
from sklearn.model_selection import KFold
from fastcore.foundation import L

from dies.data import (
    TimeseriesTransform,
    get_samples_per_day,
    tp_from_df_from_dtypes,
)


class SourceData:
    def __init__(
        self,
        files: list,
        include_shift_features: bool = False,
        standardize=True,
        n_folds: int = 1,
        add_y_to_x=False,
        add_x_to_y=False,
        cols_to_drop: list = [],
        batch_size=64,
        valid_percent=0.1,
        valid_idxs=None,
        set_minimal_validation=False,
        shuffle_train=True,
        old_drop_cols: bool = False,
    ) -> None:

        self.files = listify(files)
        self.include_shift_features = include_shift_features
        self.standardize = standardize
        self.n_folds = n_folds
        self.cols_to_drop = cols_to_drop
        self.add_y_to_x = add_y_to_x
        self.add_x_to_y = add_x_to_y
        self.batch_size = batch_size
        self.valid_percent = valid_percent
        self.valid_idxs = valid_idxs
        self.set_minimal_validation = set_minimal_validation
        self.shuffle_train = shuffle_train
        self.df_train, self.df_test, self.files = get_all_dfs(
            files,
            include_shift_features=include_shift_features,
            standardize=self.standardize,
            old_drop_cols=old_drop_cols,
        )

        self._set_n_samples_per_day()

        self._drop_cols()
        self._set_minimal_validation()

        self._create_tabular_pandas()

    def _set_minimal_validation(self):
        if self.set_minimal_validation:
            p = 0.05
            if self.n_train_samples * p <= 1:
                p = 0.1
            if self.n_train_samples * p <= 1:
                p = 0.2
            if self.n_train_samples * p <= 1:
                p = 0.3
            if self.n_train_samples * p <= 1:
                p = 0.4
            if self.n_train_samples * p <= 1:
                p = 0.5

            self.valid_percent = p

    def _drop_cols(self):
        if self.df_train is not None:
            self.df_train.drop(self.cols_to_drop, axis=1, inplace=True)
        if self.df_test is not None:
            self.df_test.drop(self.cols_to_drop, axis=1, inplace=True)

    def _create_tabular_pandas(self, categorify=True):
        self.tp_train = L()
        self.tp_test = None
        if self.df_train is not None:
            if self.n_folds > 1:
                kf = KFold(n_splits=self.n_folds)
                for _, val_idx in kf.split(range(len(self.df_train))):

                    self.tp_train += tp_from_df_from_dtypes(
                        self.df_train.drop("TestFlag", axis=1, errors="ignore"),
                        "PowerGeneration",
                        val_idxs=val_idx,
                        standardize_X=False,
                        do_split_by_n_weeks=False,
                        add_y_to_x=self.add_y_to_x,
                        add_x_to_y=self.add_x_to_y,
                        categorify=categorify,
                    )
            else:
                if self.valid_idxs is not None:
                    valid_percent = None
                else:
                    valid_percent = self.valid_percent

                self.tp_train += tp_from_df_from_dtypes(
                    self.df_train.drop("TestFlag", axis=1, errors="ignore"),
                    "PowerGeneration",
                    valid_percent=valid_percent,
                    standardize_X=False,
                    do_split_by_n_weeks=False,
                    val_idxs=self.valid_idxs,
                    add_y_to_x=self.add_y_to_x,
                    add_x_to_y=self.add_x_to_y,
                    categorify=categorify,
                )

            self.tp_test = self.tp_train[0].new(
                self.df_test.drop("TestFlag", axis=1, errors="ignore")
            )
            self.tp_test.process()

    @property
    def n_train_samples(self):
        if self.df_train is not None:
            return self.df_train.shape[0]
        else:
            return 0

    @property
    def n_test_samples(self):
        if self.df_test is not None:
            return self.df_test.shape[0]
        else:
            return 0

    @property
    def n_input_features(self):
        """
        Return the amount of continuous features in the TabularPandas.
        Returns
        -------
        integer
            amount of continuous features in the TabularPandas
        """
        if self.tp_train is not None:
            return len(self.tp_train[0].cont_names)
        else:
            return 0

    # @property
    def _set_n_samples_per_day(self):
        """
        Sets the number of samples per day.
        Returns
        -------
        integer
            amount of samples within a day
        """
        if self.df_train is not None:
            self.n_samples_per_day = get_samples_per_day(self.df_train)
        else:
            self.n_samples_per_day = 0

    def _get_bs(self, tp):
        if self.batch_size > len(tp):
            return len(tp) // 4
        else:
            return self.batch_size

    @property
    def train_dataloaders(self):
        return L(
            tp.dataloaders(
                bs=self._get_bs(tp),
                shuffle_train=self.shuffle_train,
                drop_last=True,
            )
            for tp in self.tp_train
        )

    @property
    def test_dataloader(self):
        return self.tp_test.dataloaders(
            bs=self.batch_size * 4,
            shuffle_train=False,
            drop_last=False,
        )

    @property
    def categorical_dimensions(self):
        categorical_dimensions = L()

        for c in self.tp_train[0].cat_names:
            cat_dim = self.tp_train[0][c].max()  # + 2
            if c == "Month" and cat_dim != 12:
                cat_dim = 12
            elif c == "Hour" and cat_dim != 24:
                cat_dim = 24

            categorical_dimensions += cat_dim + 1

        return categorical_dimensions

    # @property
    # def test_dataloader(self):
    #     return self.train_dataloaders[0].test_dl(self.df_test)
    # dataloaders(bs=self.batch_size, shuffle=False, drop_last=False)

    # can be done with dl = learn.dls.test_dl(test_df)


class TargetData(SourceData):
    def __init__(
        self,
        files: list,
        include_shift_features: bool = False,
        standardize=True,
        n_folds: int = 1,
        add_y_to_x=False,
        add_x_to_y=False,
        cols_to_drop: list = [],
        batch_size=64,
        task_id=None,
        split_date=None,
        num_days_training=7,
        set_minimal_validation=True,
        valid_percent=0.1,
        valid_idxs=None,
        shuffle_train=True,
        months_to_train_on=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        auxilary_df=None,
        old_drop_cols: bool = False,
        minimum_training_fraction=0.5,
    ):
        super(TargetData, self).__init__(
            files=files,
            include_shift_features=include_shift_features,
            standardize=standardize,
            n_folds=n_folds,
            add_y_to_x=add_y_to_x,
            add_x_to_y=add_x_to_y,
            cols_to_drop=cols_to_drop,
            batch_size=batch_size,
            set_minimal_validation=False,
            valid_percent=None,
            shuffle_train=shuffle_train,
            old_drop_cols=old_drop_cols,
            valid_idxs=valid_idxs,
        )
        self.months_to_train_on = months_to_train_on
        self.num_days_training = num_days_training
        self.update_task_id(task_id)

        if self.df_train is not None:
            self._set_split_date(split_date)

            self._split_for_target()
            if (
                self.df_train is not None
                and len(self.df_train)
                > (num_days_training * self.n_samples_per_day)
                * minimum_training_fraction
            ):
                if auxilary_df is not None:
                    df_aux = copy.copy(self.df_train)
                    df_aux.PowerGeneration = auxilary_df.Yhat
                    self.df_train = pd.concat([self.df_train, df_aux], axis=0)

                self.valid_percent = valid_percent
                self.set_minimal_validation = set_minimal_validation
                self._set_minimal_validation()
                self._create_tabular_pandas(categorify=False)

    def update_task_id(self, task_id):
        self.task_id = task_id
        self.df_train.TaskID = task_id
        self.df_test.TaskID = task_id

    def _set_split_date(self, split_date):
        if split_date is None:
            self.split_date = str(self.df_test.index[0].year)
        else:
            self.split_date = split_date

        self.split_date = pd.to_datetime(f"{self.split_date}-01-01", utc=True)

    def _split_for_target(self):
        df = pd.concat([copy.copy(self.df_train), copy.copy(self.df_test)], axis=0)

        if self.task_id is not None:
            df.TaskID = self.task_id

        df_train, df_test = split_in_train_and_test(df)
        # mask = df.index < self.split_date

        # self.df_train, self.df_test = df[mask], df[~mask]

        # #  make sure to have only year of test data
        # test_year = self.df_test.index[0].year + 1
        # mask = self.df_test.index < pd.to_datetime(f"{test_year}-01-01", utc=True)
        # self.df_test = self.df_test[mask]

        mask = self.df_train.index.month.isin(self.months_to_train_on)
        self.df_train = self.df_train[mask]

        self.df_train = self.df_train[
            -(self.n_samples_per_day * self.num_days_training) :
        ]


class SourceDataTimeSeries(SourceData):
    def __init__(
        self,
        files: list,
        include_shift_features: bool = False,
        standardize=True,
        n_folds: int = 1,
        add_y_to_x=False,
        add_x_to_y=False,
        cols_to_drop: list = [],
        batch_size=64,
        valid_percent=0.1,
        valid_idxs=None,
        set_minimal_validation=False,
        shuffle_train=True,
        old_drop_cols: bool = False,
    ):
        super(SourceDataTimeSeries, self).__init__(
            files=files,
            include_shift_features=include_shift_features,
            standardize=standardize,
            n_folds=n_folds,
            add_y_to_x=add_y_to_x,
            add_x_to_y=add_x_to_y,
            cols_to_drop=cols_to_drop,
            batch_size=batch_size,
            valid_percent=None,
            valid_idxs=None,
            set_minimal_validation=False,
            shuffle_train=shuffle_train,
            old_drop_cols=old_drop_cols,
        )
        self.ts_train = None
        self.ts_test = None

        if valid_idxs is not None:
            raise NotImplementedError(
                "Valid idx for timeseries currently not supported."
            )

        if self.df_train is not None:
            self.valid_percent = valid_percent
            self.set_minimal_validation = set_minimal_validation
            self._create_timeseries()
            self._set_minimal_validation()

    @property
    def n_train_samples(self):
        if self.df_train is not None:
            return len(self.ts_train[0])
        else:
            return 0

    @property
    def n_test_samples(self):
        return len(self.ts_test)

    def _create_timeseries(self):
        self.ts_train = L(
            TimeseriesTransform(
                tp_train,
                timeseries_length=self.n_samples_per_day,
                step_size=self.n_samples_per_day,
                batch_first=True,
                sequence_last=True,
                is_train=False,
                drop_inconsistent_cats=True,
            )
            for tp_train in self.tp_train
        )
        if self.tp_test is not None:
            self.ts_test = TimeseriesTransform(
                self.tp_test,
                timeseries_length=self.n_samples_per_day,
                step_size=self.n_samples_per_day,
                batch_first=True,
                sequence_last=True,
                is_train=False,
                drop_inconsistent_cats=True,
            )

    @property
    def train_dataloaders(self):
        if len(self.ts_train) == 1:
            splits = None
            if self.valid_percent > 0:
                splits = RandomSplitter(valid_pct=self.valid_percent)
            tls = self.ts_train[0].to_tfmd_lists(splits=splits)
            dls = L(
                tls.dataloaders(
                    bs=self.batch_size,
                    shuffle_train=self.shuffle_train,
                    drop_last=False,
                )
            )
        else:
            dls = L(
                ts_train.to_tfmd_lists().dataloaders(
                    bs=self.batch_size,
                    shuffle_train=self.shuffle_train,
                    drop_last=False,
                )
                for ts_train in self.ts_train
            )

        return dls

    @property
    def test_dataloaders(self):
        return self.ts_test.dataloaders(
            bs=self.batch_size * 4,
            shuffle_train=False,
            drop_last=False,
        )


class TargetDataTimeSeries(TargetData):
    def __init__(
        self,
        files: list,
        include_shift_features: bool = False,
        standardize=True,
        n_folds: int = 1,
        add_y_to_x=False,
        add_x_to_y=False,
        cols_to_drop: list = [],
        batch_size=64,
        task_id=None,
        split_date=None,
        valid_idxs=None,
        num_days_training=7,
        valid_percent=0.1,
        set_minimal_validation=True,
        shuffle_train=True,
        months_to_train_on=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        auxilary_df=None,
        old_drop_cols: bool = False,
        minimum_training_fraction=0.5,
    ):

        super(TargetDataTimeSeries, self).__init__(
            files=files,
            include_shift_features=include_shift_features,
            standardize=standardize,
            n_folds=n_folds,
            add_y_to_x=add_y_to_x,
            add_x_to_y=add_x_to_y,
            cols_to_drop=cols_to_drop,
            batch_size=batch_size,
            task_id=task_id,
            split_date=split_date,
            num_days_training=num_days_training,
            valid_percent=None,
            set_minimal_validation=False,
            shuffle_train=shuffle_train,
            months_to_train_on=months_to_train_on,
            auxilary_df=auxilary_df,
            old_drop_cols=old_drop_cols,
            minimum_training_fraction=minimum_training_fraction,
        )
        self.valid_percent = valid_percent
        self.set_minimal_validation = set_minimal_validation
        self.ts_train = None
        self.ts_test = None

        if (
            self.df_train is not None
            and len(self.df_train)
            > (num_days_training * self.n_samples_per_day) * minimum_training_fraction
        ):
            self._create_timeseries()
            self._set_minimal_validation()

    @property
    def n_train_samples(self):
        if self.ts_train is not None:
            return len(self.ts_train[0])
        else:
            return 0

    def _create_timeseries(self):
        self.ts_train = L(
            TimeseriesTransform(
                tp_train,
                timeseries_length=self.n_samples_per_day,
                step_size=self.n_samples_per_day,
                batch_first=True,
                sequence_last=True,
                is_train=False,
                drop_inconsistent_cats=True,
            )
            for tp_train in self.tp_train
        )

        self.ts_test = TimeseriesTransform(
            self.tp_test,
            timeseries_length=self.n_samples_per_day,
            step_size=self.n_samples_per_day,
            batch_first=True,
            sequence_last=True,
            is_train=False,
            drop_inconsistent_cats=True,
        )

    @property
    def train_dataloaders(self):
        if self.n_train_samples < self.batch_size:
            self.batch_size = self.n_train_samples // 4

        if self.ts_train is not None and len(self.ts_train) == 1:
            splits = None
            if self.valid_percent > 0:
                splits = RandomSplitter(valid_pct=self.valid_percent)

            tls = self.ts_train[0].to_tfmd_lists(splits=splits)
            dls = L(
                tls.dataloaders(
                    bs=self.batch_size,
                    shuffle_train=self.shuffle_train,
                    drop_last=False,
                )
            )
        elif self.ts_train is not None:
            dls = L(
                ts_train.to_tfmd_lists().dataloaders(
                    bs=self.batch_size,
                    shuffle_train=self.shuffle_train,
                    drop_last=False,
                )
                for ts_train in self.ts_train
            )
        else:
            dls = L()

        return dls

    @property
    def test_dataloader(self):
        tls = self.ts_test.to_tfmd_lists()
        dls = tls.dataloaders(
            bs=self.batch_size * 4, shuffle_train=False, drop_last=False
        )

        return dls
