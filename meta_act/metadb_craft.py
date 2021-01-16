import inspect
import logging
import math
import traceback
from typing import Union, Generator, List

import numpy as np
import pandas as pd
import tsfel
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier

from meta_act.act_learner import ActiveLearner
from meta_act.util import dict_str_creator
from meta_act.windows import get_window_features
from meta_act.windows import get_windows


def create_metadb(stream_files: Union[Generator, List[str]], z_vals_n: int,
                  z_val_selection_margin=0.02,
                  window_pre_train_sample_n=300,
                  window_adwin_delta=0.0001,
                  window_hf_kwargs=None,
                  z_val_hf_kwargs=None,
                  mfe_features=None,
                  tsfel_config=None,
                  features_summaries=None,
                  output_path=None,
                  stop_conditions=None,
                  max_failures=100):
    pre_train_sample_amount = window_pre_train_sample_n

    logging.info("--Meta-Database Creation Parameters--\n")

    logging.info(f"Z-Value Candidates Amount: {z_vals_n}.")
    logging.info(f"Z-Value Selection Margin: {z_val_selection_margin}.")
    logging.info(f"Prequential Eval Pre-train samples (Window Detection): "
                 f"{window_pre_train_sample_n}.")
    logging.info(f"Adwin Delta (Window Detection): {window_adwin_delta}.")

    if window_hf_kwargs is None:
        window_hf_kwargs = {}
        log_txt = "Default values"
    else:
        log_txt = dict_str_creator(window_hf_kwargs)
    logging.info(f"Hoeffding Tree (Window detection) hyperparameters: "
                 f"{log_txt}.")
    pre_train_sample_amount += window_hf_kwargs.get("grace_period", 200)

    if z_val_hf_kwargs is None:
        z_val_hf_kwargs = {}
        log_txt = "Default values"
    else:
        log_txt = dict_str_creator(z_val_hf_kwargs)
    logging.info(f"Hoeffding Tree (Z-val selection) hyperparameters: "
                 f"{log_txt}.")
    pre_train_sample_amount += z_val_hf_kwargs.get("grace_period", 200)

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
            z_vals = generate_zvals(n_classes, z_vals_n)
            logging.info(f"{n_classes} classes found in {stream_file_name},"
                         f"z_vals= {', '.join([str(z) for z in z_vals])}")
            windows = get_windows(full_stream_file, window_pre_train_sample_n,
                                  window_adwin_delta, window_hf_kwargs)
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
                        hits, miss, accuracy, query = learner.next_data()
                        results.append({"acc": accuracy, "query": query})
                    stream_results[str(z_val)] = results
                    stream.restart()

                # Best Z-val
                z, acc, queries, top_zs = get_best_z_last(
                    stream_results, selection_margin=z_val_selection_margin
                )

                # MetaDB features
                stream_feats = get_window_features(stream_npX,
                                                   mfe_features,
                                                   tsfel_config,
                                                   features_summaries,
                                                   n_classes)

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


def generate_zvals(classes, n_vals, log_func=None):
    if log_func is None:
        log_func = print
    max_entropy = math.log(classes, 2)
    interval = round(max_entropy / n_vals, 2)
    log_func(f"Max_entropy={max_entropy}, interval set to {interval}")
    z_vals = [round(x, 2) for x in np.arange(interval, max_entropy, interval)]
    return z_vals


def get_best_z_last(data, selection_margin=0.01):
    pds = {"z": [], "last_acc": [], "queries": []}
    for z, dat in data.items():
        z_df = pd.DataFrame(dat)
        last_acc = z_df.iloc[-1]["acc"]
        queries = (z_df["query"] * 1).sum()
        pds["z"].append(z)
        pds["last_acc"].append(last_acc)
        pds["queries"].append(queries)
    all_df = pd.DataFrame(pds)
    max_acc = all_df["last_acc"].max()
    top = all_df.loc[all_df["last_acc"] >= max_acc - selection_margin]
    ideal_z = top.sort_values(by=["queries"]).iloc[0]
    return ideal_z["z"], ideal_z["last_acc"], ideal_z["queries"], top.shape[0]
