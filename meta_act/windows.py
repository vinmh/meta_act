import logging

from skmultiflow.drift_detection import ADWIN

from meta_act.learner import get_error_hoeffdingtree


def get_windows(data, pre_train_size, delta, hf_kwargs):
    eval_data = get_error_hoeffdingtree(data, pre_train_size,
                                        **hf_kwargs)
    windows = adwin_windows(eval_data, delta, index_start=pre_train_size)
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
