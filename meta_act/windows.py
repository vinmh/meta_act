from skmultiflow.drift_detection import ADWIN


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


def fixed_windows(data, windows_n=None, windows_size=None):
    if windows_n is None and windows_size is None:
        raise ValueError("Either number of windows or size must be set")

    if windows_n is not None:
        windows_size = len(data) // windows_n
    windows = [(i, i + windows_size) for i in range(0, len(data), windows_size)]

    return windows
