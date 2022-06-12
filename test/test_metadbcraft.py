import sys

import numpy as np
import pandas as pd

from meta_act.ds_gen import DatasetGeneratorPackage, DatasetGenerator
from meta_act.metadb_craft import (
    MetaDBCrafter, WindowResults, ZValGenerator, StreamGenerator
)


class MockGenerator:
    def __init__(self, hp1, hp2, hp3):
        self.hp1 = hp1
        self.hp2 = hp2
        self.hp3 = hp3
        self.has_samples = True

    def has_more_samples(self):
        return self.has_samples

    def next_sample(self):
        return np.array([[1 for _ in range(self.hp1)]]), np.array([self.hp2])


class MockZValGenerator(ZValGenerator):
    def __init__(self):
        super().__init__(0)

    def generate_z_vals(self, class_n: int):
        return [x * 0.05 for x in range(1, class_n + 1)]

    def get_best_z(self, data: WindowResults):
        return data.get_results_table()[0]


class MockModel:
    def __init__(self):
        self.samples = 0

    def partial_fit(self, X, y, classes):
        self.samples += X.shape[0]

    def predict(self, X):
        return X[:, -1]

    def predict_proba(self, X):
        return [0.5, 0.5]


class MockDriftDetector():
    def __init__(self):
        self.samples = 0

    def add_element(self, X):
        self.samples += 1

    def detected_change(self):
        return self.samples % 10 == 0 if self.samples > 0 else False


class MockStreamGenerator(StreamGenerator):
    def __init__(self):
        generators = [
            DatasetGenerator(
                "MockGenerator",
                {"hp1": [1, 2, 3], "hp2": [4, 5, 6], "hp3": [7, 8, 9]},
                sys.modules[__name__]
            ),
            DatasetGenerator(
                "MockGenerator",
                {"hp1": [1, 10, 20], "hp2": [30, 40, 50], "hp3": [60, 70, 80]},
                sys.modules[__name__]
            )
        ]

        package = DatasetGeneratorPackage(generators, 100)
        super().__init__(MockZValGenerator(), package)
        for i in range(10):
            ds = package.generate_dataset(i)
            self.queue.put(ds)


def test_window_results_summaries():
    results = WindowResults()
    results.add_result("0.1", 0.8, True)
    results.add_result("0.1", 0.9, False)
    results.add_result("0.1", 0.6, True)

    assert results.get_last_acc("0.1") == 0.6
    assert results.get_max_acc("0.1") == 0.9
    assert results.get_mean_acc("0.1") == (0.8 + 0.9 + 0.6) / 3
    assert results.get_query_count("0.1") == 2


def test_window_results_get_results_table():
    results = WindowResults()
    results.add_result("0.1", 0.8, True)
    results.add_result("0.1", 0.9, False)
    results.add_result("0.1", 0.6, True)
    results.add_result("0.2", 0.7, True)
    results.add_result("0.2", 0.6, True)
    results.add_result("0.2", 0.9, True)
    results.add_result("0.3", 0.4, False)
    results.add_result("0.2", 0.7, True)

    table = results.get_results_table()
    print(table)
    sorted_table = table[np.argsort(table[:, 1])]
    print(sorted_table)

    assert table.shape == (3, 3)
    assert sorted_table[0][0] == 0.3
    assert sorted_table[1][0] == 0.1
    assert sorted_table[2][0] == 0.2
    assert sorted_table[0][1] == 0.4
    assert sorted_table[1][1] == 0.6
    assert sorted_table[2][1] == 0.7
    assert sorted_table[0][2] == 0
    assert sorted_table[1][2] == 2
    assert sorted_table[2][2] == 4


def test_generate_zvals():
    gen1 = ZValGenerator(4, threshold=None)
    gen2 = ZValGenerator(4, threshold=(0.05, 0.9))

    z_vals1 = gen1.generate_z_vals(4)
    z_vals2 = gen2.generate_z_vals(4)

    print(z_vals1)
    print(z_vals2)

    expected_z_vals1 = [0.4, 0.8, 1.2, 1.6]
    expected_z_vals2 = [0.05, 0.33, 0.61, 0.89]

    assert all([x in z_vals1 for x in expected_z_vals1])
    assert all([x in z_vals2 for x in expected_z_vals2])


def test_get_best_z():
    gen1 = ZValGenerator(4, selection_margin=0.02)
    gen2 = ZValGenerator(4, selection_margin=0.5)

    data = WindowResults()
    data.add_result("0.1", 0.5, True)
    data.add_result("0.1", 0.7, False)
    data.add_result("0.1", 0.6, True)
    data.add_result("0.2", 0.65, True)
    data.add_result("0.2", 0.65, True)
    data.add_result("0.2", 0.65, True)

    assert gen1.get_best_z(data)[0] == 0.2
    assert gen2.get_best_z(data)[0] == 0.1


def test_stream_generator(datadir):
    generators = [
        DatasetGenerator(
            "MockGenerator",
            {"hp1": [1, 2, 3], "hp2": [4, 5, 6], "hp3": [7, 8, 9]},
            sys.modules[__name__]
        ),
        DatasetGenerator(
            "MockGenerator",
            {"hp1": [0, 10, 20], "hp2": [30, 40, 50], "hp3": [60, 70, 80]},
            sys.modules[__name__]
        )
    ]
    package = DatasetGeneratorPackage(generators, 10)

    z_val_gen = ZValGenerator(4)
    stream_gen = StreamGenerator(z_val_gen, package, datadir / "test_cache")
    stream_gen.start_generator()

    ds_cnt = 3
    ds_list = []
    for _ in range(ds_cnt):
        ds, name = stream_gen.queue.get(timeout=30)
        print(name)
        ds_list.append((ds, name))

    stream_gen.signal_stop()
    stream_gen.wait_generator_stop()

    assert len(ds_list) == 3
    for i, ds in enumerate(ds_list):
        print(ds[1])
        print(ds[0])
        if i == 0:
            assert ds[1] == "elecNormNew.npy"
            assert ds[0].shape == (45312, 8)
        else:
            assert ds[1] != "elecNormNew.npy"
            assert ds[0].shape[0] == 10
    assert stream_gen.process is None


def test_partition_target():
    crafter = MetaDBCrafter(MockZValGenerator(), threads_number=4)
    targets = crafter.partition_target(2026)

    assert targets[0] == 507
    assert targets[1] == 507
    assert targets[2] == 506
    assert targets[3] == 506


def test_create_samples(datadir):
    crafter = MetaDBCrafter(
        MockZValGenerator(), MockModel, {}, MockDriftDetector,
        {}, 5, 5, ["min", "max"], None, ["mean"]
    )

    samples = crafter.create_samples(MockStreamGenerator(), 10)

    assert samples.shape == (10, 6)


def test_create_metadb(datadir):
    crafter = MetaDBCrafter(
        MockZValGenerator(), MockModel, {}, MockDriftDetector,
        {}, 5, 5, ["min", "max"], None, ["mean"], datadir / "metadb.csv",
        2
    )

    metadb = crafter.create_metadb(MockStreamGenerator(), 20)

    assert metadb.shape[0] == 20
    assert (datadir / "metadb.csv").exists()
    assert pd.read_csv(datadir / "metadb.csv").shape == (20, 6)
