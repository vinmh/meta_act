import pandas as pd
from skmultiflow.data.data_stream import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier

from meta_act.supervised_learner import SupervisedLearner

METADB_PATH = "./test/data/elecNormNew.csv"


def test_streaming():
    df = pd.read_csv(METADB_PATH)
    stream = DataStream(df)

    learner = SupervisedLearner(HoeffdingTreeClassifier())

    for i in range(1000):
        X, y = stream.next_sample()
        learner.next_data(X, y, stream.target_values)

    assert learner.accuracy > 0
    assert learner.hits >= 0
    assert learner.miss >= 0
    assert learner.samples_seen == 1000
