from statistics import mean

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


def test_active_learning_window_extraction_with_delta():
    df = pd.read_csv(METADB_PATH)
    stream = DataStream(df)

    learner = ActiveLearner(0.1, stream, HoeffdingTreeClassifier(),
                            store_history=True)

    for i in range(1000):
        learner.next_data()

    new_curr1 = mean([x[2] for x in learner.history])
    old_last_window_acc1 = learner.last_window_acc
    expected_delta1 = new_curr1 - old_last_window_acc1
    wind1 = learner.get_last_window(delta_acc_summary_func="mean")

    for i in range(1000):
        learner.next_data()

    new_curr2 = max([x[2] for x in learner.history])
    old_last_window_acc2 = learner.last_window_acc
    expected_delta2 = new_curr2 - old_last_window_acc2
    wind2 = learner.get_last_window(n_classes=5, delta_acc_summary_func="max")

    print(wind1)
    print(wind2)

    assert wind1.shape[0] == 1
    assert wind1.shape[1] > 0
    assert wind2.shape[0] == 1
    assert wind2.shape[1] > 0
    assert expected_delta1 == wind1["window_acc_delta"].to_numpy()[0]
    assert expected_delta2 == wind2["window_acc_delta"].to_numpy()[0]
    assert old_last_window_acc2 == new_curr1
