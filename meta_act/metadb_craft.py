import math
import shutil
from collections import defaultdict
from multiprocessing import Process, Queue, Value
from pathlib import Path
from queue import Empty as QueueEmptyError

import numpy as np
import pandas as pd
from skmultiflow.data import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees import HoeffdingTreeClassifier

from meta_act.act_learner import ActiveLearner
from meta_act.ds_gen import DatasetGeneratorPackage
from meta_act.windows import get_window_features, drift_windows


class WindowResults:
    SUMMARY_LAST = 0
    SUMMARY_MAX = 1
    SUMMARY_MEAN = 2

    def __init__(self):
        self._results = defaultdict(list)

    def add_result(self, z_val: str, accuracy: float, query: bool):
        self._results[z_val].append((accuracy, query))

    @property
    def z_vals(self):
        return list(self._results.keys())

    @property
    def has_results(self):
        return len(self.z_vals) > 0

    def get_results_table(self, summary_func=SUMMARY_LAST):
        if summary_func == self.SUMMARY_LAST:
            summary = self.get_last_acc
        elif summary_func == self.SUMMARY_MAX:
            summary = self.get_max_acc
        elif summary_func == self.SUMMARY_MEAN:
            summary = self.get_mean_acc
        else:
            raise ValueError("Invalid summary function")

        results = np.array(
            [[float(z), summary(z), self.get_query_count(z)] for z in
             self.z_vals]
        )
        return results

    def get_last_acc(self, z_val: str):
        return self._results[z_val][-1][0]

    def get_max_acc(self, z_val: str):
        return max(x[0] for x in self._results[z_val])

    def get_mean_acc(self, z_val: str):
        return np.mean([x[0] for x in self._results[z_val]])

    def get_query_count(self, z_val: str):
        return sum([int(x[1]) for x in self._results[z_val]])


class ZValGenerator:
    def __init__(
            self,
            z_val_n,
            selection_margin=0.02,
            threshold=None,
            ranking_summary_function=WindowResults.SUMMARY_LAST
    ):
        self.z_val_n = z_val_n
        self.selection_margin = selection_margin
        self.threshold = threshold
        self.summary_func = ranking_summary_function

    def generate_z_vals(self, class_n: int):
        if self.threshold is not None:
            min_value = self.threshold[0]
            max_value = self.threshold[1]
            interval = round((max_value - min_value) / (self.z_val_n - 1), 2)
        else:
            max_value = math.log(class_n, 2)
            interval = round(max_value / (self.z_val_n + 1), 2)
            min_value = interval

        z_vals = [round(x, 2)
                  for x in np.arange(min_value, max_value, interval)]

        return z_vals

    def get_best_z(self, data: WindowResults):
        results_table = data.get_results_table(self.summary_func)
        max_acc = np.max(results_table[:, 1])
        top_z = results_table[
            np.where(results_table[:, 1] >= max_acc - self.selection_margin)
        ]
        ideal_z = top_z[np.argmin(top_z[:, 2])]
        return ideal_z


class StreamGenerator:
    def __init__(
            self,
            generator_package: DatasetGeneratorPackage,
            cache_dir="./.window_cache"
    ):
        self.generator_package = generator_package
        self.queue = Queue()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stop = Value('i', 0)
        self.process = None

    def reset_cache(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def provide_streams(self):
        stream_n = 0
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        existing_files = list(self.cache_dir.glob("*.npy"))

        while self.stop.value == 0:
            if self.queue.empty():
                if stream_n < len(existing_files):
                    s_data = np.load(existing_files[stream_n])
                    s_name = existing_files[stream_n].name
                else:
                    s_data, s_name = self.generator_package.generate_dataset(
                        stream_n
                    )
                    s_name = s_name + ".npy"
                self.queue.put((s_data, s_name))
                stream_n += 1
        print("Stream generator stopped!")

    def start_generator(self):
        print("Starting up Stream generator...")
        self.process = Process(target=self.provide_streams)
        self.process.start()

    def signal_stop(self):
        self.stop.value = 1

    def wait_generator_stop(self):
        if self.stop:
            self.process.join()
            self.process = None


class MetaDBCrafter:

    def __init__(
            self,
            z_val_generator: ZValGenerator,
            classifier=None,
            classifier_kwargs=None,
            drift_detector=None,
            drift_detector_kwargs=None,
            wind_gen_grace_period=300,
            act_learner_grace_period=200,
            mfe_features=None,
            tsfel_config=None,
            feature_summaries=None,
            output_path=None,
            threads_number=4,
    ):
        self.z_val_generator = z_val_generator
        self.act_learner_grace_period = act_learner_grace_period
        self.wind_gen_grace_period = wind_gen_grace_period
        self.mfe_features = mfe_features
        self.tsfel_config = tsfel_config
        if feature_summaries is None:
            self.feature_summaries = ["max", "min", "mean", "var"]
        else:
            self.feature_summaries = feature_summaries
        self.output_path = output_path
        self.thread_n = threads_number
        if classifier is None:
            self.classifier = HoeffdingTreeClassifier
            self.classifier_kwargs = {}
        else:
            self.classifier = classifier
            if classifier_kwargs is None:
                self.classifier_kwargs = {}
            else:
                self.classifier_kwargs = classifier_kwargs
        if drift_detector is None:
            self.drift_detector = ADWIN
            self.drift_detector_kwargs = {"delta": 0.0001}
        else:
            self.drift_detector = drift_detector
            if drift_detector_kwargs is None:
                self.drift_detector_kwargs = {}
            else:
                self.drift_detector_kwargs = drift_detector_kwargs

    def create_samples(
            self, stream_generator: StreamGenerator, target: int
    ):
        samples = []
        try:
            while len(samples) < target:
                stream_file, stream_name = stream_generator.queue.get(
                    timeout=120
                )

                n_classes = np.unique(stream_file[:, -1].astype(int)).shape[0]
                z_vals = self.z_val_generator.generate_z_vals(n_classes)

                stream_data = DataStream(
                    stream_file[:, :-1], stream_file[:, -1].astype(int)
                )
                drift_detector = self.drift_detector(
                    **self.drift_detector_kwargs)
                wind_classifier = self.classifier(**self.classifier_kwargs)
                data_X = []
                data_y = []

                for X, y, change in drift_windows(
                        stream_data, drift_detector, wind_classifier,
                        self.wind_gen_grace_period
                ):
                    data_X.append(X)
                    data_y.append(y)
                    if change:
                        window_X = np.concatenate(data_X)
                        window_y = np.concatenate(data_y)
                        wind_results = WindowResults()

                        pre_X = window_X[:self.act_learner_grace_period]
                        pre_y = window_y[:self.act_learner_grace_period]
                        rest_X = window_X[self.act_learner_grace_period:]
                        rest_y = window_y[self.act_learner_grace_period:]

                        for z in z_vals:
                            act_model = self.classifier(
                                **self.classifier_kwargs)
                            learner = ActiveLearner(z, act_model)

                            learner.prequential_eval(
                                pre_X, pre_y, stream_data.target_values
                            )
                            for X, y in zip(rest_X, rest_y):
                                hits, miss, acc, query, _ = learner.next_data(
                                    np.array([X]), y, stream_data.target_values
                                )
                                wind_results.add_result(str(z), acc, query)

                        if wind_results.has_results:
                            best_z, acc, queries = \
                                self.z_val_generator.get_best_z(
                                    wind_results
                                )

                            features = get_window_features(
                                window_X, self.mfe_features, self.tsfel_config,
                                self.feature_summaries
                            )

                            features["dataset_name"] = stream_name
                            features["actl_acc"] = acc
                            features["actl_queries"] = queries

                            features["Y"] = float(best_z)

                            samples.append(features)

                    if len(samples) >= target:
                        break

        except QueueEmptyError:
            print("Stream Generator timeout...")
        return pd.concat(samples).reset_index(drop=True)

    def _creator_thread_func(self, stream_generator, target, queue):
        samples = self.create_samples(stream_generator, target)
        queue.put(samples)

    def partition_target(self, target):
        targets = [target // self.thread_n for _ in range(self.thread_n)]
        for r in range(target % self.thread_n):
            targets[r % len(targets)] += 1
        return targets

    def create_metadb(
            self, stream_generator: StreamGenerator, target=2026
    ):

        target_per_thread = self.partition_target(target)

        results_queue = Queue()
        processes = [
            Process(
                target=self._creator_thread_func,
                args=(stream_generator, target_t, results_queue)
            )
            for target_t in target_per_thread
        ]

        stream_generator.start_generator()

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        stream_generator.signal_stop()

        metadb_chunks = []
        while not results_queue.empty():
            metadb_chunks.append(results_queue.get())

        full_metadb = pd.concat(metadb_chunks).reset_index(drop=True)
        if self.output_path is not None:
            full_metadb.to_csv(self.output_path, index=False)

        return full_metadb
