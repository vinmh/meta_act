import shutil
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


class MethodRunner:
    def __init__(self, classifier, classifier_kwargs):
        self.classifier = classifier
        if classifier_kwargs is not None:
            self.classifier_kwargs = classifier_kwargs
        else:
            self.classifier_kwargs = {}

    def init(self):
        pass

    def pretrain(self, X, y, target_values):
        pass

    def next_data(self, sample_n, dataset_name, X, y, target_values, save_data):
        return None


class EmptyRunner(MethodRunner):
    def __init__(self):
        super().__init__(None, None)


class ZMTLRunner(MethodRunner):
    def __init__(
            self,
            results_handler,
            mfe_features,
            tsfel_config,
            feature_summaries,
            meta_learner,
            classifier=HoeffdingTreeClassifier,
            classifier_kwargs=None,
            drift_detector=ADWIN,
            drift_detector_kwargs=None,
            initial_z_val=0.5,
    ):
        self.mfe_features = mfe_features
        self.tsfel_config = tsfel_config
        self.feature_summaries = feature_summaries
        self.meta_learner = meta_learner
        self.z_val = initial_z_val
        self.results_handler = results_handler
        self.queries = 0
        self.drift_detector = drift_detector
        if drift_detector_kwargs is not None:
            self.drift_detector_kwargs = drift_detector_kwargs
        else:
            self.drift_detector_kwargs = {}
        super().__init__(classifier, classifier_kwargs)
        self.learner = None
        self.drift_model = None

    def init(self):
        model = self.classifier(**self.classifier_kwargs)
        self.learner = ActiveLearner(self.z_val, model, store_history=True)
        self.drift_model = self.drift_detector(**self.drift_detector_kwargs)

    def pretrain(self, X, y, target_values):
        if self.learner is None:
            return

        self.learner.prequential_eval(
            X, y, target_values
        )

    def next_data(self, sample_n, dataset_name, X, y, target_values, save_data):
        if self.learner is None or self.drift_model is None:
            return None

        data = self.learner.next_data(X, y, target_values)
        self.queries += int(data[3])
        if save_data:
            self.results_handler.write_zmtl(
                sample_n, dataset_name, self.z_val, data[2], self.queries
            )

        self.drift_model.add_element(int(data[4]))
        if self.drift_model.detected_change():
            features = self.learner.get_last_window(
                self.mfe_features, self.tsfel_config, self.feature_summaries
            )
            new_z_val = self.meta_learner.predict(features)
            print(f"New Z-val generated: {new_z_val}")
            self.z_val = new_z_val[0]
            self.learner.z_val = self.z_val

        return data


class FixedZRunner(MethodRunner):
    def __init__(
            self,
            results_handler,
            z_vals: List[float],
            classifier=HoeffdingTreeClassifier,
            classifier_kwargs=None
    ):
        self.z_vals = z_vals
        self.results_handler = results_handler
        super().__init__(classifier, classifier_kwargs)
        self.classifier_models = []
        self.queries = defaultdict(lambda: 0)

    def init(self):
        self.classifier_models = [
            ActiveLearner(z_val, self.classifier(**self.classifier_kwargs))
            for z_val in self.z_vals
        ]

    def pretrain(self, X, y, target_values):
        if len(self.classifier_models) == 0:
            return

        for learner in self.classifier_models:
            learner.prequential_eval(
                X, y, target_values
            )

    def next_data(self, sample_n, dataset_name, X, y, target_values, save_data):
        if len(self.classifier_models) == 0:
            return None

        results = []
        for learner in self.classifier_models:
            data = learner.next_data(X, y, target_values)
            self.queries[learner.z_val] += int(data[3])
            if save_data:
                self.results_handler.write_fixed_z(
                    sample_n, dataset_name, learner.z_val, data[2],
                    self.queries[learner.z_val]
                )

            results.append(data)

        return results


class SupervisedRunner(MethodRunner):
    def __init__(
            self,
            results_handler,
            classifier=HoeffdingTreeClassifier,
            classifier_kwargs=None
    ):
        self.results_handler = results_handler
        super().__init__(classifier, classifier_kwargs)
        self.model = None

    def init(self):
        self.model = SupervisedLearner(
            self.classifier(**self.classifier_kwargs)
        )

    def pretrain(self, X, y, target_values):
        if self.model is None:
            return

        self.model.prequential_eval(X, y, target_values)

    def next_data(self, sample_n, dataset_name, X, y, target_values, save_data):
        if self.model is None:
            return None

        data = self.model.next_data(X, y, target_values)
        if save_data:
            self.results_handler.write_supervised(
                sample_n, dataset_name, data[2]
            )

        return data


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
        self.write_supervised("sample", "dataset_name", "acc")

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


def join_partial_files(file_list):
    type_dict = defaultdict(list)
    for file in file_list:
        type_name = file.name.rsplit(".", 2)[-1]
        type_dict[type_name].append(file)
    for file_type, files in type_dict.items():
        concat_file = pd.concat([
            pd.read_csv(x) for x in files
        ]).reset_index(drop=True)
        out_filename = files[0].name.split(".", 2)[0] + f"_{file_type}.csv"
        concat_file.to_csv(files[0].parent / out_filename, index=False)


class MetaLearnerManager:
    def __init__(
            self,
            meta_learner_regressor=RandomForestRegressor,
            meta_learner_kwargs=None,
            train_metadb_minority_threshold=100,
            train_metadb_oversample=True,
            train_metadb_scale=True,
            train_metadb_metadata_size=5
    ):
        self.meta_learner_regressor = meta_learner_regressor
        if meta_learner_kwargs is None:
            self.meta_learner_kwargs = {}
        else:
            self.meta_learner_kwargs = meta_learner_kwargs
        self.metadb_minority = train_metadb_minority_threshold
        self.metadb_oversample = train_metadb_oversample
        self.metadb_scale = train_metadb_scale
        self.metadb_metadata_size = train_metadb_metadata_size

    def train_metamodel(self, metadb: pd.DataFrame):
        X = metadb.iloc[:, :-self.metadb_metadata_size]
        y = metadb.Y

        learner = MetaLearner(self.meta_learner_regressor,
                              **self.meta_learner_kwargs)
        results = learner.fit(
            X, y, scale=self.metadb_scale, oversample=self.metadb_oversample,
            eliminate_minority=self.metadb_minority is not None,
            minority_threshold=self.metadb_minority
        )

        return learner, results

    def test_metamodel(self, metadb: pd.DataFrame, folds=10):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)

        X = metadb.iloc[:, :-self.metadb_metadata_size]
        y = metadb.Y.astype(str)

        cv_pairs = [
            (X.iloc[train_index], y.iloc[train_index].astype(float),
             X.iloc[test_index], y.iloc[test_index].astype(float),)
            for train_index, test_index in skf.split(X, y)
        ]

        results = []
        for X_train, y_train, X_test, y_test in cv_pairs:
            learner = MetaLearner(self.meta_learner_regressor,
                                  **self.meta_learner_kwargs)
            pair_results = learner.fit(
                X_train, y_train, test_data=(X_test, y_test)
            )
            results.append(pair_results)

        results_df = pd.DataFrame(results).mean()
        return results_df


class ExperimentRunner:
    def __init__(
            self,
            benchmark_dataset_dir="./benchmark",
            classifier=HoeffdingTreeClassifier,
            classifier_kwargs=None,
            drift_detector=ADWIN,
            drift_detector_kwargs=None,
            use_fixed_z_vals=True,
            use_supervised=True,
            fixed_z_vals=None,
            experiments_threads=2,
            experiments_tsfel_config=None,
            experiments_mfe_features=None,
            experiments_feature_summaries=None,
            experiments_grace_period=200,
            experiments_results_save_delay=500
    ):
        self.dataset_dir = Path(benchmark_dataset_dir)
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
        self.threads = experiments_threads
        if experiments_feature_summaries is None:
            self.feature_summaries = ["max", "min", "mean", "var"]
        else:
            self.feature_summaries = experiments_feature_summaries
        self.mfe_features = experiments_mfe_features
        self.tsfel_config = experiments_tsfel_config
        self.grace_period = experiments_grace_period
        self.result_delay = experiments_results_save_delay

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
            print(f"Starting up {dataset}")
            dataset_df = pd.read_csv(dataset)
            dataset_name = dataset.name

            sample_n = 0
            zmtl = ZMTLRunner(
                results_handler, self.mfe_features, self.tsfel_config,
                self.feature_summaries, learner, self.classifier,
                self.classifier_kwargs, self.drift_detector,
                self.drift_detector_kwargs, 0.5
            )

            if self.use_fixed_z_vals:
                fixed_z = FixedZRunner(
                    results_handler, self.fixed_z_vals, self.classifier,
                    self.classifier_kwargs
                )
            else:
                fixed_z = EmptyRunner()

            if self.use_supervised:
                supervised = SupervisedRunner(
                    results_handler, self.classifier, self.classifier_kwargs
                )
            else:
                supervised = EmptyRunner()

            stream = DataStream(dataset_df)

            zmtl.init()
            fixed_z.init()
            supervised.init()

            pretrain_X, pretrain_y = stream.next_sample(self.grace_period)
            zmtl.pretrain(pretrain_X, pretrain_y, stream.target_values)
            fixed_z.pretrain(pretrain_X, pretrain_y, stream.target_values)
            supervised.pretrain(pretrain_X, pretrain_y, stream.target_values)

            while stream.has_more_samples():
                X, y = stream.next_sample()
                sample_n += 1
                save_data = sample_n % self.result_delay == 0

                zmtl.next_data(
                    sample_n, dataset_name, X, y, stream.target_values,
                    save_data
                )
                fixed_z.next_data(
                    sample_n, dataset_name, X, y, stream.target_values,
                    save_data
                )
                supervised.next_data(
                    sample_n, dataset_name, X, y, stream.target_values,
                    save_data
                )

        results_handler.close_all()

    def run_experiments(self, learner: MetaLearner, out_path: Union[str, Path]):
        benchmark_dataset_list = list(self.dataset_dir.glob("*.csv"))
        datasets_per_thread = self._partition_datasets_per_thread(
            benchmark_dataset_list
        )

        shutil.rmtree(Path(out_path), ignore_errors=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

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

        partial_results_files = list(
            Path(out_path).parent.glob(f"{Path(out_path.name)}.*.*")
        )
        join_partial_files(partial_results_files)
