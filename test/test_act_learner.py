import pandas as pd
from skmultiflow.data.data_stream import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier

from meta_act.act_learner import ActiveLearner

METADB_PATH = "./test/data/elecNormNew.csv"


def test_active_learning_window_extraction():
    df = pd.read_csv(METADB_PATH)
    stream = DataStream(df)

    learner = ActiveLearner(0.1, stream, HoeffdingTreeClassifier(),
                            store_history=True)

    for i in range(1000):
        learner.next_data()

    wind1 = learner.get_last_window()

    for i in range(1000):
        learner.next_data()

    wind2 = learner.get_last_window(n_classes=5)

    print(wind1)
    print(wind2)

    assert wind1.shape[0] == 1
    assert wind1.shape[1] > 0
    assert wind2.shape[0] == 1
    assert wind2.shape[1] > 0
