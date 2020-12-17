from act_meta.learner import get_error_hoeffdingtree
from act_meta.windows import adwin_windows, fixed_windows


def get_windows(data_file, method, **kwargs):
    if method == "adwin":
        pre_train_size = kwargs["pre_train_size"]
        tree_kwargs = kwargs["hf_kwargs"]
        delta = kwargs["delta"]
        eval_data = get_error_hoeffdingtree(data_file, pre_train_size, **tree_kwargs)
        windows = adwin_windows(eval_data, delta, index_start=pre_train_size)
        print(f"{len(windows)} adwin windows for file {data_file}")
        return windows
    elif method == "fixed":
        windows_n = kwargs["windows_n"] if "windows_n" in kwargs else None
        windows_size = kwargs["windows_size"] if "windows_size" in kwargs else None
        with open(data_file, "r") as f:
            data = list(f)
        windows = fixed_windows(data, windows_n, windows_size)
        print(f"{len(windows)} fixed windows for file {data_file}")
        return windows
