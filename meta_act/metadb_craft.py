import inspect
import json
import logging
import math
import shutil
import traceback
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Union, Generator, List

import numpy as np
import pandas as pd
import tsfel
from skmultiflow.data import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees import HoeffdingTreeClassifier

from meta_act.act_learner import ActiveLearner
from meta_act.ds_gen import DatasetGeneratorPackage
from meta_act.util import dict_str_creator
from meta_act.windows import get_window_features
from meta_act.windows import old_get_windows, get_windows


class MetaDBCrafter:
    WINDOW_METADATA_FILENAME = "window_metadata.json"

    def __init__(
            self,
            z_vals_n,
            z_val_selection_margin=0.02,
            fixed_z_val_threshold=None,
            window_pre_train_sample_n=300,
            window_drift_detector=None,
            window_drift_detector_kwargs=None,
            window_classifier=None,
            window_classifier_kwargs=None,
            window_acc_summary="max",
            mfe_features=None,
            tsfel_config=None,
            feature_summaries=None,
            output_path=None,
            target_samples=200000,
            thread_package_size=10,
            threads_number=4,
            use_window_cache=True,
            cache_dir="./.window_cache"
    ):
        self.z_vals_n = z_vals_n
        self.z_val_selection_margin = z_val_selection_margin
        self.fixed_z_val_threshold = fixed_z_val_threshold
        self.window_pre_train_size = window_pre_train_sample_n
        if window_drift_detector is None:
            self.window_drift_detector = ADWIN(0.0001)
        else:
            self.window_drift_detector = window_drift_detector
            if window_drift_detector_kwargs is None:
                self.window_drift_detector_kwargs = {}
            else:
                self.window_drift_detector_kwargs = window_drift_detector_kwargs
        if window_classifier is None:
            self.window_classifier = HoeffdingTreeClassifier()
        else:
            self.window_classifier = window_classifier
            if window_classifier_kwargs is None:
                self.window_classifier_kwargs = {}
            else:
                self.window_classifier_kwargs = window_classifier_kwargs
        self.window_acc_summary = window_acc_summary
        self.mfe_features = mfe_features
        self.tsfel_config = tsfel_config
        if feature_summaries is None:
            self.feature_summaries = ["max", "min", "mean", "var"]
        else:
            self.feature_summaries = feature_summaries
        self.output_path = output_path
        self.target_samples = target_samples
        self.thread_packages = thread_package_size
        self.thread_n = threads_number
        self.use_window_cache = use_window_cache
        self.cache_dir = Path(cache_dir)

    def _fetch_metadata(self):
        metadata_file = self.cache_dir / self.WINDOW_METADATA_FILENAME
        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            metadata = json.load(f)

        return metadata

    def _fetch_cache(self):
        if not self.cache_dir.exists():
            return None

        metadata = self._fetch_metadata()
        if metadata is None:
            return None

        window_filelist = list(x.name for x in self.cache_dir.glob("*.npz"))
        if len(window_filelist) == 0:
            return None

        return window_filelist, metadata

    def _generate_zvals(self, classes):
        if self.fixed_z_val_threshold is not None:
            min_value = self.fixed_z_val_threshold[0]
            max_value = self.fixed_z_val_threshold[1]
            interval = round((max_value - min_value) / self.z_vals_n, 2)
        else:
            max_value = math.log(classes, 2)
            interval = round(max_value / self.z_vals_n, 2)
            min_value = interval

        z_vals = [round(x + min_value, 2)
                  for x in np.arange(min_value, max_value, interval)]

        return z_vals

    def _get_best_z(self, data):
        pds = {"z": [], "acc": [], "queries": []}
        for z, dat in data.items():
            z_df = pd.DataFrame(dat)
            if self.window_acc_summary == "max":
                acc = z_df.iloc[-1]["acc"]
            else:
                acc = z_df.mean()["acc"]
            queries = (z_df["query"] * 1).sum()
            pds["z"].append(z)
            pds["acc"].append(acc)
            pds["queries"].append(queries)
        all_df = pd.DataFrame(pds)
        max_acc = all_df["acc"].max()
        top = all_df.loc[all_df["acc"] >= max_acc - self.z_val_selection_margin]
        ideal_z = top.sort_values(by=["queries"]).iloc[0]
        return ideal_z["z"], ideal_z["acc"], ideal_z["queries"], top.shape[0]

    def generate_windows(
            self, generator_package: DatasetGeneratorPackage,
    ):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True)

        windows_written = 0
        files_list = []
        stream_metadata = defaultdict(dict)
        target_reached = False
        for stream_file, stream_name in generator_package.generate_datasets():
            drift_detector = self.window_drift_detector(
                **self.window_drift_detector_kwargs
            )
            classifier = self.window_classifier(
                **self.window_classifier_kwargs
            )
            windows = get_windows(stream_file,
                                  self.window_pre_train_size,
                                  drift_detector,
                                  classifier)
            n_classes = np.unique(stream_file[:, -1].astype(int)).shape[0]
            z_vals = self._generate_zvals(n_classes)
            stream_metadata[stream_name]["z_vals"] = z_vals
            stream_metadata[stream_name]["windows"] = {}

            file_count = 0
            file_data = []
            for window_n, window in enumerate(windows):
                if windows_written >= self.target_samples:
                    target_reached = True
                    break

                windows_written += 1
                stream_metadata[stream_name]["windows"][str(window_n)] = {
                    "start": window[0], "end": window[1], "classes": n_classes
                }
                win_data = stream_file[window[0]: window[1]]
                file_data.append([window_n, win_data])
                if (len(file_data) >= self.thread_packages or
                        windows_written >= self.target_samples):
                    filename = (
                        f"{file_count}__{len(file_data)}__{stream_name}.npz"
                    )
                    file_dict = {
                        f"{n}": data
                        for n, data in file_data
                    }
                    np.savez(
                        (self.cache_dir / filename).as_posix(), **file_dict
                    )
                    files_list.append(filename)

            if target_reached:
                break

        with open(self.cache_dir / self.WINDOW_METADATA_FILENAME) as f:
            json.dump(stream_metadata, f)

        return files_list

    def parse_window_file(self, filename, metadata):
        file_path = Path(self.cache_dir) / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing specified window file: {filename}"
            )

        window_n, window_qtd, stream_name = filename.split("__")
        window_file = np.load(file_path)
        window_list = window_file["files"]
        z_vals = metadata[stream_name]["z_vals"]

        metadb_samples = []
        for window_name in window_list:
            window_metadata = metadata[stream_name]["windows"][window_name]
            window_data = window_file[window_name]
            window_x = window_data[:, :-1]
            window_y = window_data[:, -1].astype(int)

            stream = DataStream(window_x, y=window_y)

            wind_results = {}
            for z_val in z_vals:
                model = self.window_classifier(**self.window_classifier_kwargs)
                learner = ActiveLearner(z_val, stream, model)
                results = []
                query = False
                while query is not None:
                    hits, miss, acc, query, _ = learner.next_data()
                    results.append({"acc": acc, "query": query})
                wind_results[str(z_val)] = results
                stream.reset()

            z, acc, queries, top_z = self._get_best_z(wind_results)

            features = get_window_features(
                window_x, self.mfe_features, self.tsfel_config,
                self.feature_summaries, n_classes=window_metadata["classes"]
            )

            features["dataset_name"] = stream_name
            features["actl_acc"] = acc
            features["actl_queries"] = queries
            features["window_start"] = (
                metadata[stream_name]["windows"][window_name]["start"]
            )
            features["window_end"] = (
                metadata[stream_name]["windows"][window_name]["end"]
            )
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

        return files_per_thread

    def create_metadb_threaded(
            self, generator_package: DatasetGeneratorPackage
    ):
        if not self.use_window_cache:
            file_list = self.generate_windows(generator_package)
            metadata = self._fetch_metadata()
        else:
            cache = self._fetch_cache()
            if cache is None:
                file_list = self.generate_windows(generator_package)
                metadata = self._fetch_metadata()
            else:
                file_list = cache[0]
                metadata = cache[1]

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
            metadb_chunks.append(results_queue.get())

        full_metadb = pd.concat(metadb_chunks)
        if self.output_path is not None:
            full_metadb.to_csv(self.output_path)

        return full_metadb


def create_metadb(stream_files: Union[Generator, List[str]], z_vals_n: int,
                  z_val_selection_margin=0.02,
                  fixed_z_val_threshold=None,
                  window_pre_train_sample_n=300,
                  window_adwin_delta=0.0001,
                  window_hf_kwargs=None,
                  z_val_hf_kwargs=None,
                  wind_acc_summary="max",
                  fixed_windows_size=None,
                  mfe_features=None,
                  tsfel_config=None,
                  features_summaries=None,
                  window_acc_delta_feature=True,
                  output_path=None,
                  stop_conditions=None,
                  max_failures=100):
    pre_train_sample_amount = window_pre_train_sample_n

    logging.info("--Meta-Database Creation Parameters--\n")

    logging.info(f"Z-Value Candidates Amount: {z_vals_n}.")
    logging.info(f"Z-Value Selection Margin: {z_val_selection_margin}.")
    if fixed_z_val_threshold is None or not isinstance(fixed_z_val_threshold,
                                                       tuple):
        logging.info("Z-Value Fixed selection threshold not set.")
    else:
        l_thresh = fixed_z_val_threshold[0]
        u_thresh = fixed_z_val_threshold[1]
        logging.info(f"Z-Value Fixed selection threshold set to:"
                     f" {l_thresh} <= x <= {u_thresh}.")
    logging.info(f"Prequential Eval Pre-train samples (Window Detection): "
                 f"{window_pre_train_sample_n}.")
    if fixed_windows_size is not None:
        logging.info(f"Using fixed windows with size: {fixed_windows_size}")
        use_fixed_windows = True
        window_adwin_delta = fixed_windows_size
    else:
        logging.info("Generating windows through Adwin")
        logging.info(f"Adwin Delta (Window Detection): {window_adwin_delta}.")
        use_fixed_windows = False

    if not use_fixed_windows:
        if window_hf_kwargs is None:
            window_hf_kwargs = {}
            log_txt = "Default values"
        else:
            log_txt = dict_str_creator(window_hf_kwargs)
        logging.info(f"Hoeffding Tree (Window detection) hyperparameters: "
                     f"{log_txt}.")
        pre_train_sample_amount += window_hf_kwargs.get("grace_period", 200)
    else:
        window_hf_kwargs = {}

    if z_val_hf_kwargs is None:
        z_val_hf_kwargs = {}
        log_txt = "Default values"
    else:
        log_txt = dict_str_creator(z_val_hf_kwargs)
    logging.info(f"Hoeffding Tree (Z-val selection) hyperparameters: "
                 f"{log_txt}.")
    pre_train_sample_amount += z_val_hf_kwargs.get("grace_period", 200)

    logging.info(f"Accuracy summary used for z-val and window features:"
                 f" {wind_acc_summary}")

    if mfe_features is None:
        mfe_features = ["nr_class", "attr_ent", "kurtosis", "skewness"]
        log_txt = f"(Default): {', '.join(mfe_features)}"
    else:
        log_txt = f": {', '.join(mfe_features)}"
    logging.info(f"MetaDatabase MFE features {log_txt}.")

    if tsfel_config is None:
        tsfel_config = tsfel.get_features_by_domain()
        log_txt = f"(Default): All"
    else:
        log_txt = f": {tsfel_config}"
    logging.info(f"Metadatabase TSFEL features {log_txt}.")

    if features_summaries is None:
        features_summaries = ["max", "min", "mean", "var"]
        log_txt = f"(Default): {', '.join(features_summaries)}"
    else:
        log_txt = f": {', '.join(features_summaries)}"
    logging.info(f"Metadatabase feature summarizations {log_txt}")

    logging.info(f"Include Mean Window Accuracy Delta Feature:"
                 f" {window_acc_delta_feature}")

    if inspect.isgenerator(stream_files):
        stream_from_generator = True
        log_txt = "on-the-fly"
    else:
        stream_from_generator = False
        log_txt = ', '.join(stream_files)
    logging.info(f"Stream files: {log_txt}.")

    if output_path is None:
        logging.info("No output path specified, returning metadatabase in-"
                     "memory.")
    else:
        logging.info(f"Metadatabase will be saved at: {output_path}.")

    logging.info(f"About {pre_train_sample_amount} samples from each stream "
                 f"dataset will be lost to pre-training procedures.")
    if stop_conditions is None:
        if stream_from_generator:
            logging.warning("Stream files from generator, but no stop "
                            "conditions set, if the generator is infinite, "
                            "this will run FOREVER!")
        else:
            logging.info("No stopping conditions set, running until all "
                         "stream datasets are exhausted.")
        stop_conditions = {}
    else:
        logging.info(
            f"Stopping conditions set: {dict_str_creator(stop_conditions)}")
    logging.info("-------------------------------------")

    metadb = None
    failures = 0
    stream_files_used = 0
    last_window_acc = 0
    for stream_file in stream_files:
        stream_file_name = "<Error in getting stream name>"
        try:
            if not stream_from_generator:
                full_stream_file = np.genfromtxt(stream_file, delimiter=",")[1:]
                stream_file_name = stream_file
            else:
                full_stream_file = stream_file[0]
                stream_file_name = stream_file[1]
            full_stream_fileX = full_stream_file[:, :-1]
            full_stream_fileY = full_stream_file[:, -1].astype(int)
            n_classes = np.unique(full_stream_fileY).shape[0]
            z_vals = generate_zvals(n_classes, z_vals_n,
                                    fixed_thresh=fixed_z_val_threshold)
            logging.info(f"{n_classes} classes found in {stream_file_name},"
                         f"z_vals= {', '.join([str(z) for z in z_vals])}")
            windows = old_get_windows(full_stream_file,
                                      window_pre_train_sample_n,
                                      window_adwin_delta, window_hf_kwargs,
                                      use_fixed_windows=use_fixed_windows)
            window_grace_period = window_hf_kwargs.get("grace_period", 200)
            windows = list(
                filter(lambda x: x[1] - x[0] >= window_grace_period * 2,
                       windows)
            )

            for window_n, window in enumerate(windows):
                # Stream results
                stream_npX = full_stream_fileX[window[0]: window[1]]
                stream_npY = full_stream_fileY[window[0]: window[1]]
                stream = DataStream(stream_npX, y=stream_npY)
                stream_results = {}
                for z_val in z_vals:
                    hf = HoeffdingTreeClassifier(**z_val_hf_kwargs)
                    learner = ActiveLearner(z_val, stream, hf)
                    results = []
                    query = False
                    while query is not None:
                        hits, miss, accuracy, query, _ = learner.next_data()
                        results.append({"acc": accuracy, "query": query})
                    stream_results[str(z_val)] = results
                    stream.restart()

                # Best Z-val
                z, acc, queries, top_zs = get_best_z_last(
                    stream_results, selection_margin=z_val_selection_margin,
                    acc_param=wind_acc_summary
                )

                if window_acc_delta_feature:
                    last_acc = last_window_acc
                    curr_acc = acc
                    last_window_acc = acc
                else:
                    last_acc = None
                    curr_acc = None

                # MetaDB features
                stream_feats = get_window_features(stream_npX,
                                                   mfe_features,
                                                   tsfel_config,
                                                   features_summaries,
                                                   n_classes,
                                                   last_acc,
                                                   curr_acc)

                # Metadata
                stream_feats["dataset_name"] = stream_file_name
                stream_feats["window_method"] = "adwin"
                stream_feats["window_hp"] = window_adwin_delta
                stream_feats["actl_algorithm"] = "hftree"
                stream_feats["actl_acc"] = acc
                stream_feats["actl_queries"] = queries
                stream_feats["window_start"] = window[0]
                stream_feats["window_end"] = window[1]

                # Z-Val
                stream_feats["Y"] = float(z)

                if metadb is None:
                    metadb = stream_feats
                else:
                    metadb = pd.concat([metadb, stream_feats]).reset_index(
                        drop=True)
            stream_files_used += 1
        except Exception as e:
            logging.error(f"{stream_file_name} failed, reason: {e}")
            traceback.print_exc()
            failures += 1
            if failures >= max_failures:
                break
        finally:
            stop_running = False
            if metadb is not None:
                for k, v in stop_conditions.items():
                    if k == "max_datasets":
                        if stream_files_used > v:
                            stop_running = True
                            break
                    if k == "max_samples":
                        if metadb.shape[0] > v:
                            stop_running = True
                            break
                    if k == "minority_target":
                        y_cnt = metadb["Y"].value_counts(
                            ascending=True).iloc[0]
                        if y_cnt >= v:
                            stop_running = True
                            break
                    if k == "majority_target":
                        y_cnt = metadb["Y"].value_counts(
                            ascending=False).iloc[0]
                        if y_cnt >= v:
                            stop_running = True
                            break
                if stop_running:
                    break

    if output_path is None:
        return metadb
    else:
        metadb.to_csv(output_path, index=False)
        return True


def generate_zvals(classes, n_vals, log_func=None, fixed_thresh=None):
    if log_func is None:
        log_func = print
    if fixed_thresh is not None:
        min_value = fixed_thresh[0]
        max_value = fixed_thresh[1]
        interval = round((max_value - min_value) / n_vals, 2)
        log_func(f"Z-val between [{min_value}, {max_value}], interval set"
                 f"to {interval}")
    else:
        max_value = math.log(classes, 2)
        interval = round(max_value / n_vals, 2)
        min_value = interval
        log_func(f"Max_entropy={max_value}, interval set to {interval}")
    z_vals = [round(x + min_value, 2)
              for x in np.arange(min_value, max_value, interval)]
    return z_vals


def get_best_z_last(data, selection_margin=0.01, acc_param="max"):
    pds = {"z": [], "acc": [], "queries": []}
    for z, dat in data.items():
        z_df = pd.DataFrame(dat)
        if acc_param == "max":
            acc = z_df.iloc[-1]["acc"]
        else:
            acc = z_df.mean()["acc"]
        queries = (z_df["query"] * 1).sum()
        pds["z"].append(z)
        pds["acc"].append(acc)
        pds["queries"].append(queries)
    all_df = pd.DataFrame(pds)
    max_acc = all_df["acc"].max()
    top = all_df.loc[all_df["acc"] >= max_acc - selection_margin]
    ideal_z = top.sort_values(by=["queries"]).iloc[0]
    return ideal_z["z"], ideal_z["acc"], ideal_z["queries"], top.shape[0]
