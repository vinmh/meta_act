import math

import pandas as pd
from pymfe.mfe import MFE
from skmultiflow.data import DataStream
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

from meta_act.learner import get_error_classifier
from meta_act.tsfel_ext import gen_tsfel_features


def drift_windows(
        data: DataStream, drift_detector: BaseDriftDetector,
        classifier, pre_train_size
):
    for error, X, y in get_error_classifier(data, pre_train_size, classifier):
        drift_detector.add_element(error)
        yield X, y, drift_detector.detected_change()


def get_window_features(X, mfe_features, tsfel_config,
                        summary_funcs, n_classes=None,
                        last_window_acc=None, current_acc=None):
    feat_dfs = []
    if mfe_features is not None:
        mfe = MFE(features=mfe_features, summary=summary_funcs)
        mfe.fit(X)
        mfe_feats = mfe.extract()
        feat_dfs.append(pd.DataFrame(
            {name: [value] for name, value in zip(mfe_feats[0], mfe_feats[1])}
        ))

    if tsfel_config is not None:
        tsfel_feats = gen_tsfel_features(tsfel_config,
                                         pd.DataFrame(X),
                                         summary=summary_funcs)
        feat_dfs.append(tsfel_feats)

    if len(feat_dfs) > 0:
        stream_feats = pd.concat(feat_dfs, axis=1)
    else:
        stream_feats = pd.DataFrame()

    if last_window_acc is not None and current_acc is not None:
        stream_feats["window_acc_delta"] = pd.Series(
            current_acc - last_window_acc
        )

    if n_classes is not None:
        stream_feats["n_classes"] = pd.Series(n_classes)
        stream_feats["max_possible_entropy"] = pd.Series(math.log(n_classes, 2))

    return stream_feats
