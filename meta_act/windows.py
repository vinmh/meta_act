import logging
import math

import pandas as pd
from pymfe.mfe import MFE
from skmultiflow.drift_detection import ADWIN

from meta_act.learner import get_error_hoeffdingtree
from meta_act.tsfel_ext import gen_tsfel_features


def get_windows(data, pre_train_size, delta_or_windsize, hf_kwargs=None,
                use_fixed_windows=False):
    if use_fixed_windows:
        windows = fixed_windows(data, delta_or_windsize, pre_train_size)
        logging.info(f"{len(windows)} fixed windows found")
        return windows
    else:
        if hf_kwargs is None:
            hf_kwargs = {}
        eval_data = get_error_hoeffdingtree(data, pre_train_size, **hf_kwargs)
        windows = adwin_windows(eval_data, delta_or_windsize,
                                index_start=pre_train_size)
        logging.info(f"{len(windows)} adwin windows found")
        return windows


def adwin_windows(data, delta, index_start=0):
    adwin = ADWIN(delta)
    windows = []
    last_i = index_start
    for i, d in enumerate(data):
        adwin.add_element(d)
        if adwin.detected_change():
            windows.append((last_i, i + index_start))
            last_i = i + index_start
    return windows


def fixed_windows(data, size, index_start=0):
    windows = []
    last_i = index_start
    for i in range(index_start, len(data), size):
        windows.append((last_i, i + index_start))
        last_i = i + index_start
    return windows


def get_window_features(X, mfe_features, tsfel_config,
                        summary_funcs, n_classes=None,
                        last_window_acc=None, current_acc=None):
    mfe = MFE(features=mfe_features, summary=summary_funcs)
    mfe.fit(X)
    mfe_feats = mfe.extract()

    tsfel_feats = gen_tsfel_features(tsfel_config,
                                     pd.DataFrame(X),
                                     summary=summary_funcs)

    stream_feats = pd.DataFrame(
        {name: [value] for name, value in zip(mfe_feats[0], mfe_feats[1])}
    )
    stream_feats = pd.concat([stream_feats, tsfel_feats], axis=1)

    if last_window_acc is not None and current_acc is not None:
        stream_feats["window_acc_delta"] = current_acc - last_window_acc

    if n_classes is not None:
        stream_feats["n_classes"] = n_classes
        stream_feats["max_possible_entropy"] = math.log(n_classes, 2)

    return stream_feats
