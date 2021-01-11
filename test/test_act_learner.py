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

    wind = learner.get_last_window()

    print(wind)

    assert wind.shape[0] == 1
    assert wind.shape[1] > 0
