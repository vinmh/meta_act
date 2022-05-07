from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier


def get_error_classifier(data, pre_train_size, model):
    orig_X = data[:, :-1]
    orig_y = data[:, -1].astype(int)
    stream = DataStream(orig_X, orig_y)

    pretrain_X, pretrain_y = stream.next_sample(pre_train_size)

    # Pre-train
    model.partial_fit(pretrain_X, pretrain_y, classes=stream.target_values)

    evaluations = []
    while stream.has_more_samples():
        X, y = stream.next_sample()

        # Evaluation
        y_hat = model.predict(X)
        evaluations.append(int(y_hat[0] == y[0]))

        # Train
        model.partial_fit(X, y, classes=stream.target_values)

    return evaluations


def get_error_hoeffdingtree(data, pre_train_size, **hf_kwargs):
    orig_X = data[:, :-1]
    orig_y = data[:, -1].astype(int)
    stream = DataStream(orig_X, orig_y)
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
