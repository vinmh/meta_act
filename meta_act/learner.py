from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import FileStream


def get_error_hoeffdingtree(data_path, pre_train_size, **hf_kwargs):
    stream = FileStream(data_path)
    hf = HoeffdingTreeClassifier(**hf_kwargs)

    pretrainX, pretrainy = stream.next_sample(pre_train_size)

    # Pre-train
    hf.partial_fit(pretrainX, pretrainy, classes=stream.target_values)

    evaluations = []
    while stream.has_more_samples():
        X, y = stream.next_sample()

        # Evaluation
        y_hat = hf.predict(X)
        evaluations.append(int(y_hat[0] == y[0]))

        # Train
        hf.partial_fit(X, y, classes=stream.target_values)

    return evaluations
