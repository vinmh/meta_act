from act_meta import get_windows
from act_meta.tsfel_ext import gen_tsfel_features
from act_meta.act_learner import ActiveLearner
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import DataStream
import pandas as pd
import numpy as np
import queue
from pymfe.mfe import MFE
import threading
import os
import traceback
import tsfel
from functools import reduce
import math

try:
    import wandb
    print("wandb activated")
except Exception:
    print("wandb not found")

def generate_zvals(classes, n_vals, log_func=None):
    if log_func is None:
        log_func = print
    max_entropy = math.log(classes, 2)
    interval = round(max_entropy / n_vals, 2)
    log_func(f"Max_entropy={max_entropy}, interval set to {interval}")
    z_vals = [round(x,2) for x in np.arange(interval, max_entropy, interval)]
    return z_vals

class MetaDBThread(threading.Thread):
    def __init__(
        self,
        threadID,
        files,
        pre_train_size,
        hf_kwargs,
        delta,
        grace_period,
        z_vals_n,
        z_errormargin,
        meta_feats,
        tsfel_domains,
        verbose=True,
        line_replace=False,
        wandb_active=False
    ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.files = files
        self.pre_train_size = pre_train_size
        self.hf_kwargs = hf_kwargs
        self.delta = delta
        self.grace_period = grace_period
        self.z_vals_n = z_vals_n
        self.z_errormargin = z_errormargin
        self.meta_feats = meta_feats
        self.verbose = verbose
        self.verbose_prefix = ("\r" if line_replace else "\n") + f"Thread {self.threadID}: "
        self.wandb_active = wandb_active
        self.tsfel_domains = tsfel_domains
        
    def log(self, txt, verbose_check=True):
        if not verbose_check or self.verbose:
            print(f"{self.verbose_prefix}{txt}")
            if self.wandb_active:
                wandb.log({f"Thread{self.threadID}_log": txt})

    def run(self):
        local_metadb_lines = []
        for stream_file in self.files:
            self.log(f"Parsing file {stream_file}...")
            try:
                #full_stream_file = pd.read_csv(stream_file)
                full_stream_file = np.genfromtxt(stream_file, delimiter=",")[1:]
                full_stream_fileX = full_stream_file[:, :-1]
                full_stream_fileY = full_stream_file[:, -1].astype(int)
                n_classes = np.unique(full_stream_fileY).shape[0]
                z_vals = generate_zvals(n_classes, self.z_vals_n)
                self.log(f"{n_classes} classes found in {stream_file}, z_vals={z_vals}", verbose_check=False)
                windows = get_windows(
                    stream_file,
                    "adwin",
                    pre_train_size=self.pre_train_size,
                    hf_kwargs=self.hf_kwargs,
                    delta=self.delta,
                )
                windows = list(
                    filter(lambda x: x[1] - x[0] >= self.grace_period * 2, windows)
                )

                for window_n, window in enumerate(windows):
                    #Stream results
                    stream_npX = full_stream_fileX[window[0] : window[1]]
                    stream_npY = full_stream_fileY[window[0] : window[1]]
                    stream = DataStream(stream_npX, y=stream_npY)
                    stream_results = {}
                    for z_val in z_vals:
                        hf = HoeffdingTreeClassifier(grace_period=self.grace_period)
                        learner = ActiveLearner(z_val,
                                                stream,
                                                self.grace_period,
                                                hf,
                                                verbose=self.verbose,
                                                verbose_prefix=self.verbose_prefix)
                        results = []
                        query = False
                        while query is not None:
                            hits, miss, accuracy, query = learner.next_data()
                            results.append({"acc": accuracy, "query": query})
                        stream_results[str(z_val)] = results
                        stream.restart()

                    #Best Z-val
                    z, acc, queries, top_zs = get_best_z_last(
                        stream_results, error_margin=self.z_errormargin
                    )
                    self.log(f"Window {window_n+1} = Z: {z}, Mean_Acc: {acc},"
                             f" Queries: {queries}, chosen out of {top_zs} contenders")
                    
                    if self.wandb_active:
                        wandb.log({f"{stream_file}_Z": z,
                                   f"{stream_file}_MeanAcc": acc,
                                   f"{stream_file}_Queries": queries})
                    
                    #MetaDB features
                    summary = ["max", "min", "mean", "var"]
                    #MFE Features
                    mfe = MFE(features=self.meta_feats,
                              summary=summary)
                    mfe.fit(
                        stream_npX, stream_npY
                    )
                    feats = mfe.extract()
                    stream_feats = pd.DataFrame(
                        {name: [value] for name, value in zip(feats[0], feats[1])}
                    )
                    
                    #TSFEL Features
                    cfg = {k:v for k, v in tsfel.get_features_by_domain().items()
                           if k in self.tsfel_domains}
                    feats_tsfel = gen_tsfel_features(cfg, pd.DataFrame(stream_npX), summary=summary, log_func=self.log)
                    stream_feats = pd.concat([stream_feats, feats_tsfel], axis=1)
                    
                    stream_feats["dataset_name"] = stream_file
                    stream_feats["window_method"] = "adwin"
                    stream_feats["window_hp"] = self.delta
                    stream_feats["actl_algorithm"] = "hftree"
                    stream_feats["actl_acc"] = acc
                    stream_feats["actl_queries"] = queries
                    stream_feats["window_start"] = window[0]
                    stream_feats["window_end"] = window[1]
                    stream_feats["Y"] = float(z)

                    local_metadb_lines.append((stream_feats))
            except Exception as e:
                self.log(f"{stream_file} failed, reason: {e}", verbose_check=False)
                traceback.print_exc()

        self.log("Waiting for db Lock...")
        metadb_lock.acquire(blocking=True)
        metadb_queue.put(local_metadb_lines)
        metadb_lock.release()
        self.log("Finished")


metadb_lock = threading.Lock()
metadb_queue = queue.Queue()


def get_best_z_last(data, error_margin=0.01):
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
    top = all_df.loc[all_df["last_acc"] >= max_acc - error_margin]
    ideal_z = top.sort_values(by=["queries"]).iloc[0]
    return ideal_z["z"], ideal_z["last_acc"], ideal_z["queries"], top.shape[0]


def create_metadb(
    stream_files,
    threads=4,
    delta=0.0001,
    z_vals_n=5,
    hf_kwargs={},
    pre_train_size=300,
    grace_period=300,
    z_errormargin=0.02,
    meta_feats=["nr_class", "attr_ent", "class_ent", "kurtosis", "skewness"],
    tsfel_domains=["temporal"],
    output_path=None,
    verbose=True,
    line_replace=True,
    wandb_active=False
):
    
    if output_path is not None:
        if verbose:
            print("Cleaning up metadb file if it exists...")
        try:
            os.remove(output_path)
            if verbose:
                print("Done")
        except OSError:
            if verbose:
                print("File does not exists, skipping.")
        
    
    thread_file_n = [len(stream_files) // threads for _ in range(threads)]
    thread_file_n[-1] += len(stream_files) % threads
    
    previous = 0
    thread_portions = []
    for i in thread_file_n:
        thread_portions.append(stream_files[previous : previous + i])
        previous += i

    thread_insts = []
    thID = 0
    for portion in thread_portions:
        thread = MetaDBThread(
            thID,
            portion,
            pre_train_size,
            hf_kwargs,
            delta,
            grace_period,
            z_vals_n,
            z_errormargin,
            meta_feats,
            tsfel_domains,
            verbose=verbose,
            line_replace=line_replace,
            wandb_active=wandb_active
        )
        thread.start()
        thread_insts.append(thread)
        thID += 1

    for t in thread_insts:
        t.join()
    if verbose:
        print("Finishing up all threads")
        print("Merging metaDB")
    metadb = []
    while not metadb_queue.empty():
        fragment = metadb_queue.get()
        if output_path is not None:
            pd.concat(fragment).reset_index(drop=True).to_csv(output_path, mode='a', header=(not os.path.exists(output_path)))
        else:
            metadb += fragment

    if output_path is None:
        if verbose:
            print("Output path not specified, returning merged metadb dataframe instead")
        metadb = pd.concat(metadb).reset_index(drop=True)
        return metadb
    else:
        return True
