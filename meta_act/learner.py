from skmultiflow.data import DataStream


def get_error_classifier(stream: DataStream, pre_train_size: int, model):
    pretrain_X, pretrain_y = stream.next_sample(pre_train_size)

    # Pre-train
    model.partial_fit(pretrain_X, pretrain_y, classes=stream.target_values)

    while stream.has_more_samples():
        X, y = stream.next_sample()

        # Evaluation
        y_hat = model.predict(X)
        yield int(y_hat[0] == y[0]), X, y

        # Train
        model.partial_fit(X, y, classes=stream.target_values)
