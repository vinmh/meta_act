import json
import logging
import math
import shutil
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from skmultiflow.data import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees import HoeffdingTreeClassifier

from meta_act.act_learner import ActiveLearner
from meta_act.ds_gen import DatasetGeneratorPackage
from meta_act.windows import get_window_features
from meta_act.windows import get_windows


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


class WindowGenerator:
    WINDOW_METADATA_FILENAME = "window_metadata.json"

    def __init__(
            self,
            z_val_generator: ZValGenerator,
            target_windows=200000,
            pre_train_samples=300,
            window_package_size=10,
            drift_detector=None,
            drift_detector_kwargs=None,
            classifier=None,
            classifier_kwargs=None,
            cache_dir="./.window_cache",
            use_stream_cache=True
    ):
        self.z_val_generator = z_val_generator
        self.pre_train_samples = pre_train_samples
        self.target_samples = target_windows
        self.package_size = window_package_size
        if drift_detector is None:
            self.drift_detector = ADWIN
            self.drift_detector_kwargs = {"delta": 0.0001}
        else:
            self.drift_detector = drift_detector
            if drift_detector_kwargs is None:
                self.drift_detector_kwargs = {}
            else:
                self.drift_detector_kwargs = drift_detector_kwargs
        if classifier is None:
            self.classifier = HoeffdingTreeClassifier
            self.classifier_kwargs = {}
        else:
            self.classifier = classifier
            if classifier_kwargs is None:
                self.classifier_kwargs = {}
            else:
                self.classifier_kwargs = classifier_kwargs
        self.cache_dir = Path(cache_dir)
        self.use_stream_cache = use_stream_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def stream_cache_dir(self):
        return self.cache_dir / "streams"

    @property
    def window_cache_dir(self):
        return self.cache_dir / "windows"

    def _write_file(self, file_data, filename):
        file_dict = {f"{n}": data for n, data in file_data}
        np.savez(
            (self.window_cache_dir / filename).as_posix(),
            **file_dict
        )

    def get_stream(
            self, generator_package: DatasetGeneratorPackage, reset_cache=False
    ):
        if reset_cache:
            shutil.rmtree(self.stream_cache_dir)

        if self.use_stream_cache:
            if not self.stream_cache_dir.exists():
                self.stream_cache_dir.mkdir(parents=True)
            existing_files = list(self.stream_cache_dir.glob("*.npy"))
        else:
            existing_files = []

        for cache_file in existing_files:
            stream_data = np.load(cache_file)
            yield stream_data, cache_file.name

        logging.debug("-- Stream cache exhausted, generating new data --")

        for stream_file, stream_name in generator_package.generate_datasets():
            if self.use_stream_cache:
                np.save(self.stream_cache_dir / stream_name, stream_file)
            yield stream_file, stream_name + ".npy"

    def generate_windows(
            self, generator_package: DatasetGeneratorPackage,
            reset_stream_cache=False
    ):
        shutil.rmtree(self.window_cache_dir, ignore_errors=True)
        self.window_cache_dir.mkdir(parents=True)

        windows_written = 0
        files_list = []
        stream_metadata = defaultdict(dict)
        target_reached = False
        for stream_file, stream_name in self.get_stream(
                generator_package, reset_stream_cache
        ):
            drift_detector = self.drift_detector(
                **self.drift_detector_kwargs
            )
            classifier = self.classifier(
                **self.classifier_kwargs
            )
            windows = get_windows(stream_file,
                                  self.pre_train_samples,
                                  drift_detector,
                                  classifier)
            n_classes = np.unique(stream_file[:, -1].astype(int)).shape[0]
            z_vals = self.z_val_generator.generate_z_vals(n_classes)
            stream_metadata[stream_name]["z_vals"] = z_vals

            file_count = 0
            file_data = []
            for window_n, window in enumerate(windows):
                if windows_written >= self.target_samples:
                    target_reached = True
                    break

                windows_written += 1
                win_data = stream_file[window[0]: window[1]]
                file_data.append([window_n, win_data])
                if (len(file_data) >= self.package_size or
                        windows_written >= self.target_samples):
                    filename = (
                        f"{file_count}__{len(file_data)}__{stream_name}.npz"
                    )
                    self._write_file(file_data, filename)
                    files_list.append(filename)
                    file_count += 1
                    file_data = []

            if target_reached:
                break

        with open(
                self.window_cache_dir / self.WINDOW_METADATA_FILENAME, "w"
        ) as f:
            json.dump(stream_metadata, f)

        return files_list


class MetaDBCrafter:

    def __init__(
            self,
            z_val_generator: ZValGenerator,
            window_cache_dir: Path,
            classifier=None,
            classifier_kwargs=None,
            act_learner_grace_period=200,
            mfe_features=None,
            tsfel_config=None,
            feature_summaries=None,
            output_path=None,
            thread_package_size=10,
            threads_number=4,
    ):
        self.z_val_generator = z_val_generator
        self.window_cache_dir = window_cache_dir
        self.act_learner_grace_period = act_learner_grace_period
        self.mfe_features = mfe_features
        self.tsfel_config = tsfel_config
        if feature_summaries is None:
            self.feature_summaries = ["max", "min", "mean", "var"]
        else:
            self.feature_summaries = feature_summaries
        self.output_path = output_path
        self.thread_packages = thread_package_size
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

    def _fetch_window_metadata(self):
        metadata_file = (self.window_cache_dir /
                         WindowGenerator.WINDOW_METADATA_FILENAME)
        if not metadata_file.exists():
            raise FileNotFoundError(
                "Metadata file not found, re-generate windows"
            )

        with open(metadata_file) as f:
            metadata = json.load(f)

        return metadata

    def _fetch_window_cache(self):
        if not self.window_cache_dir.exists():
            raise FileNotFoundError(
                "Windows not found, please generate windows first"
            )

        metadata = self._fetch_window_metadata()

        window_filelist = list(
            x.name for x in self.window_cache_dir.glob("*.npz")
        )
        if len(window_filelist) == 0:
            raise FileNotFoundError(
                "No window files found, please generate windows"
            )

        return window_filelist, metadata

    def parse_window_file(self, filename, metadata):
        file_path = Path(self.window_cache_dir) / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing specified window file: {filename}"
            )

        window_n, window_qtd, stream_name = filename.rsplit(".", 1)[0].split(
            "__"
        )
        window_file = np.load(file_path)
        window_list = window_file.files
        z_vals = metadata[stream_name]["z_vals"]

        metadb_samples = []
        for window_name in window_list:
            window_data = window_file[window_name]
            window_x = window_data[:, :-1]
            window_y = window_data[:, -1].astype(int)

            stream = DataStream(window_x, y=window_y)

            wind_results = WindowResults()
            for z_val in z_vals:
                model = self.classifier(**self.classifier_kwargs)
                learner = ActiveLearner(z_val, model)

                pretrain_X, pretrain_y = stream.next_sample(
                    self.act_learner_grace_period
                )
                learner.prequential_eval(
                    pretrain_X, pretrain_y, stream.target_values
                )
                while stream.has_more_samples():
                    X, y = stream.next_sample()
                    hits, miss, acc, query, _ = learner.next_data(
                        X, y, stream.target_values
                    )
                    wind_results.add_result(str(z_val), acc, query)
                stream.restart()

            z, acc, queries = self.z_val_generator.get_best_z(wind_results)

            features = get_window_features(
                window_x, self.mfe_features, self.tsfel_config,
                self.feature_summaries
            )

            features["dataset_name"] = stream_name
            features["actl_acc"] = acc
            features["actl_queries"] = queries
            features["window_batch"] = window_name

            features["Y"] = float(z)

            metadb_samples.append(features)

        return pd.concat(metadb_samples).reset_index(drop=True)

    def _creator_thread_func(self, file_list, metadata, queue):
        metadb_sample_list = [
            self.parse_window_file(filename, metadata)
            for filename in file_list
        ]
        queue.put(metadb_sample_list)

    def _partition_files_per_thread(self, file_list):
        files_per_thread_n = len(file_list) // self.thread_n
        files_per_thread = [
            file_list[i:i + files_per_thread_n]
            for i in range(0, len(file_list), files_per_thread_n)
        ]
        if len(files_per_thread) > 0:
            for extra_i, extra in enumerate(files_per_thread[self.thread_n:]):
                files_per_thread[extra_i % self.thread_n] += extra

        return files_per_thread[:self.thread_n]

    def create_metadb(
            self, file_list: List[str]
    ):
        metadata = self._fetch_window_metadata()

        files_per_thread = self._partition_files_per_thread(file_list)

        results_queue = Queue()
        processes = [
            Process(
                target=self._creator_thread_func,
                args=(files, metadata, results_queue)
            )
            for files in files_per_thread
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        metadb_chunks = []
        while not results_queue.empty():
            metadb_chunks += results_queue.get()

        full_metadb = pd.concat(metadb_chunks).reset_index(drop=True)
        if self.output_path is not None:
            full_metadb.to_csv(self.output_path, index=False)

        return full_metadb
