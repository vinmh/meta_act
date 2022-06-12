import numpy as np
from skmultiflow.data import DataStream

from meta_act.learner import get_error_classifier


class MockModel:
    def __init__(self):
        self.samples = 0

    def partial_fit(self, X, y, classes):
        self.samples += X.shape[0]

    def predict(self, X):
        return X[:, -1]


def test_get_error_classifier():
    X = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    y = np.array(
        [0, 1, 1, 1]
    )
    stream = DataStream(X, y)
    model = MockModel()
    pre_train = 2

    for hit, p_X, p_y in get_error_classifier(stream, pre_train, model):
        if p_X[0, -1] == 0:
            assert hit == 0
        else:
            assert hit == 1
