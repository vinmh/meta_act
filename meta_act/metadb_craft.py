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
            windows = get_windows(full_stream_file, window_pre_train_sample_n,
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
