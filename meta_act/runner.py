import logging
from collections import defaultdict
from multiprocessing import Process
from pathlib import Path
from typing import Union, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from skmultiflow.data import DataStream
from skmultiflow.drift_detection import ADWIN
from skmultiflow.trees import HoeffdingTreeClassifier

from meta_act.act_learner import ActiveLearner
from meta_act.metalearn import MetaLearner
from meta_act.supervised_learner import SupervisedLearner


class FixedZRunner:
    def __init__(
            self,
            z_vals: List[float],
            classifier=HoeffdingTreeClassifier,
            classifier_kwargs=None
    ):
        self.z_vals = z_vals
        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs
        self.classifier_models = []

    def init(self):
        self.classifier_models = [
            ActiveLearner(z_val, self.classifier(**self.classifier_kwargs))
            for z_val in self.z_vals
        ]

    def pretrain(self, X, y, target_values):
        for learner in self.classifier_models:
            learner.prequential_eval(
                X, y, target_values
            )

    def next_data(self, X, y, target_values):
        return [
            (l.z_val, l.next_data(X, y, target_values))
            for l in self.classifier_models
        ]


class ResultsWriter:
    def __init__(self, base_path, use_fixed_z, use_supervised):
        self.base_path = base_path
        self.use_fixed_z = use_fixed_z
        self.use_supervised = use_supervised
        self.zmtl_f = None
        self.fixed_f = None
        self.super_f = None

    @property
    def zmtl_filename(self):
        return self.base_path + ".zmtl"

    @property
    def fixed_filename(self):
        return self.base_path + ".fixz"

    @property
    def supervised_filename(self):
        return self.base_path + ".trad"

    def init_files(self):
        self.zmtl_f = open(self.zmtl_filename, "w")
        if self.use_fixed_z:
            self.fixed_f = open(self.fixed_filename, "w")
        if self.use_supervised:
            self.super_f = open(self.supervised_filename, "w")

    def _write_data(
            self, handler, sample_n, dataset_name, z_val, accuracy, queries
    ):
        if handler is not None:
            data = (
                f"{sample_n},{dataset_name},{z_val},{accuracy},{queries}\n"
            )
            handler.write(data)

    def write_zmtl(self, sample_n, dataset_name, z_val, accuracy, queries):
        self._write_data(
            self.zmtl_f, sample_n, dataset_name, z_val, accuracy, queries
        )

    def write_fixed_z(self, sample_n, dataset_name, z_val, accuracy, queries):
        self._write_data(
            self.fixed_f, sample_n, dataset_name, z_val, accuracy, queries
        )

    def write_supervised(self, sample_n, dataset_name, accuracy):
        self._write_data(
            self.super_f, sample_n, dataset_name, "All Labels",
            accuracy, sample_n
        )

    def write_headers(self):
        data = ["sample", "dataset_name", "z_val", "acc", "query"]
        self.write_zmtl(*data)
        self.write_fixed_z(*data)
        self.write_supervised(*data)

    def _close(self, handler):
        if handler is not None:
            handler.close()

    def close_zmtl(self):
        self._close(self.zmtl_f)
        self.zmtl_f = None

    def close_fixed_z(self):
        self._close(self.fixed_f)
        self.fixed_f = None

    def close_supervised(self):
        self._close(self.super_f)
        self.super_f = None

    def close_all(self):
        self.close_zmtl()
        self.close_fixed_z()
        self.close_supervised()


class ExperimentRunner:
    def __init__(
            self,
            benchmark_dataset_dir="./benchmark",
            meta_learner_regressor=RandomForestRegressor,
            meta_learner_kwargs=None,
            classifier=HoeffdingTreeClassifier,
            classifier_kwargs=None,
            drift_detector=ADWIN,
            drift_detector_kwargs=None,
            use_fixed_z_vals=True,
            use_supervised=True,
            fixed_z_vals=None,
            train_metadb_minority_threshold=100,
            train_metadb_oversample=True,
            train_metadb_scale=True,
            train_metadb_metadata_size=5,
            experiments_threads=2,
            experiments_tsfel_config=None,
            experiments_mfe_features=None,
            experiments_feature_summaries=None,
            experiments_grace_period=200,
            experiments_results_save_delay=500
    ):
        self.dataset_dir = Path(benchmark_dataset_dir)
        self.meta_learner = meta_learner_regressor
        if meta_learner_kwargs is None:
            self.meta_learner_kwargs = {}
        else:
            self.meta_learner_kwargs = meta_learner_kwargs
        self.classifier = classifier
        if classifier_kwargs is None:
            self.classifier_kwargs = {}
        else:
            self.classifier_kwargs = classifier_kwargs
        self.drift_detector = drift_detector
        if drift_detector_kwargs is None:
            self.drift_detector_kwargs = {}
        else:
            self.drift_detector_kwargs = drift_detector_kwargs
        self.use_fixed_z_vals = use_fixed_z_vals
        self.use_supervised = use_supervised
        if use_fixed_z_vals:
            if fixed_z_vals is None:
                self.fixed_z_vals = [0.05, 0.1, 0.2, 0.5, 0.7]
            else:
                self.fixed_z_vals = fixed_z_vals
        else:
            self.fixed_z_vals = None
        self.metadb_minority = train_metadb_minority_threshold
        self.metadb_oversample = train_metadb_oversample
        self.metadb_scale = train_metadb_scale
        self.metadb_metadata_size = train_metadb_metadata_size
        self.threads = experiments_threads
        if experiments_feature_summaries is None:
            self.feature_summaries = ["max", "min", "mean", "var"]
        else:
            self.feature_summaries = experiments_feature_summaries
        self.mfe_features = experiments_mfe_features
        self.tsfel_config = experiments_tsfel_config
        self.grace_period = experiments_grace_period
        self.result_delay = experiments_results_save_delay

    def train_metamodel(self, metadb: pd.DataFrame):
        X = metadb.iloc[:, :-self.metadb_metadata_size]
        y = metadb.Y

        learner = MetaLearner(self.meta_learner, **self.meta_learner_kwargs)
        results = learner.fit(
            X, y, scale=self.metadb_scale, oversample=self.metadb_oversample,
            eliminate_minority=self.metadb_minority is not None,
            minority_threshold=self.metadb_minority
        )

        return learner, results

    def test_metamodel(self, metadb: pd.DataFrame, folds=10):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)

        X = metadb.iloc[:, :-self.metadb_metadata_size]
        y = metadb.Y

        cv_pairs = [
            (X.iloc[train_index], y.iloc[train_index].astype(float),
             X.iloc[test_index], y.iloc[test_index].astype(float),)
            for train_index, test_index in skf.split(X, y)
        ]

        results = []
        for X_train, y_train, X_test, y_test in cv_pairs:
            learner = MetaLearner(self.meta_learner, **self.meta_learner_kwargs)
            pair_results = learner.fit(
                X_train, y_train, test_data=(X_test, y_test)
            )
            results.append(pair_results)

        results_df = pd.DataFrame(results).groupby().mean()
        return results_df

    def _partition_datasets_per_thread(self, dataset_list):
        files_per_thread_n = len(dataset_list) // self.threads
        datasets_per_thread = [
            dataset_list[i:i + files_per_thread_n]
            for i in range(0, len(dataset_list), files_per_thread_n)
        ]
        if len(datasets_per_thread) > 0:
            for extra_i, extra in enumerate(datasets_per_thread[self.threads:]):
                datasets_per_thread[extra_i % self.threads] += extra

        return datasets_per_thread[:self.threads]

    def _experiment_thread(self, learner: MetaLearner, dataset_list, outpath):
        results_handler = ResultsWriter(
            outpath, self.use_fixed_z_vals, self.use_supervised
        )
        results_handler.init_files()
        results_handler.write_headers()
        for dataset in dataset_list:
            logging.info(f"Starting up {dataset}")
            dataset_df = pd.read_csv(dataset)
            dataset_name = dataset.name

            sample_n = 0
            z_val = 0.5
            zmtl_queries = 0
            fixed_queries = defaultdict(lambda: 0)
            stream = DataStream(dataset_df)
            drift_det = self.drift_detector(**self.drift_detector_kwargs)
            act_model = self.classifier(**self.classifier_kwargs)
            act_learner = ActiveLearner(
                z_val, act_model, store_history=True
            )

            if self.use_fixed_z_vals:
                fixed_z_runner = FixedZRunner(
                    self.fixed_z_vals, self.classifier,
                    self.classifier_kwargs
                )
                fixed_z_runner.init()
            else:
                fixed_z_runner = None

            if self.use_supervised:
                supervised_model = SupervisedLearner(
                    self.classifier(**self.classifier_kwargs)
                )
            else:
                supervised_model = None

            # Pre-train
            pretrain_X, pretrain_y = stream.next_sample(self.grace_period)
            act_learner.prequential_eval(
                pretrain_X, pretrain_y, stream.target_values
            )
            if fixed_z_runner is not None:
                fixed_z_runner.pretrain(
                    pretrain_X, pretrain_y, stream.target_values
                )
            if supervised_model is not None:
                supervised_model.prequential_eval(
                    pretrain_X, pretrain_y, stream.target_values
                )

            while stream.has_more_samples():
                X, y = stream.next_sample()
                sample_n += 1
                save_data = sample_n % self.result_delay == 0

                zmtl_data = act_learner.next_data(
                    X, y, stream.target_values
                )
                zmtl_queries += int(zmtl_data[3])
                if save_data:
                    results_handler.write_zmtl(
                        sample_n, dataset_name, z_val,
                        zmtl_data[2], zmtl_queries
                    )

                if fixed_z_runner is not None:
                    fixed_data = fixed_z_runner.next_data(
                        X, y, stream.target_values
                    )
                    for f_z_val, f_results in fixed_data:
                        fixed_queries[f_z_val] += int(f_results[3])
                        if save_data:
                            results_handler.write_fixed_z(
                                sample_n, dataset_name, f_z_val,
                                f_results[2], fixed_queries[f_z_val]
                            )

                if supervised_model is not None:
                    supervised_data = supervised_model.next_data(
                        X, y, stream.target_values
                    )
                    if save_data:
                        results_handler.write_supervised(
                            sample_n, dataset_name, supervised_data[2]
                        )

                drift_det.add_element(int(zmtl_data[4]))
                if drift_det.detected_change():
                    history_feats = act_learner.get_last_window(
                        self.mfe_features, self.tsfel_config,
                        self.feature_summaries
                    )
                    z_val = learner.predict(history_feats)[0]
                    act_learner.z_val = z_val

        results_handler.close_all()

    def run_experiments(self, learner: MetaLearner, out_path: Union[str, Path]):
        benchmark_dataset_list = list(self.dataset_dir.glob("*.csv"))
        datasets_per_thread = self._partition_datasets_per_thread(
            benchmark_dataset_list
        )

        partial_results_filenames = [
            Path(out_path).as_posix() + f".{i}"
            for i in range(self.threads)
        ]

        processes = [
            Process(
                target=self._experiment_thread,
                args=(learner, datasets, outname)
            )
            for datasets, outname in zip(
                datasets_per_thread, partial_results_filenames
            )
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        # to be continued
