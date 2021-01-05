import tsfel


def gen_tsfel_features(cfg_file, X,
                       summary=None,
                       log_func=None):
    if summary is None:
        summary = ["max", "min", "mean", "var", "std"]
    log_func = print if log_func is None else log_func
    fs = X.shape[0] * 0.2
    feats_raw = tsfel.calc_window_features(cfg_file, X, fs=fs)
    summarized = feats_raw.T.groupby(lambda x: x.split("_", 1)[1]).agg(
        summary).T
    flat = summarized.unstack().sort_index(level=1)
    flat.columns = flat.columns.map("_".join)
    found_features = flat.shape[1]
    log_func(f"Generated {found_features} tsfel features")
    return flat
