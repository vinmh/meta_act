import sys

import numpy as np

from meta_act.ds_gen import DatasetGenerator, DatasetGeneratorPackage


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


def test_generator():
    generator = DatasetGenerator(
        "MockGenerator", {"hp1": [1, 2, 3], "hp2": [4, 5, 6], "hp3": [7, 8, 9]},
        sys.modules[__name__]
    )

    assert generator.name == "Mock"
    assert generator.gen_class(1, 2, 3).hp1 == 1
    assert len(generator.random_combination().keys()) == 3


def test_generate_dataset():
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

    ds1, ds1_f = package.generate_dataset(0)
    ds2, ds2_f = package.generate_dataset(1)
    ds3, ds3_f = package.generate_dataset(2)

    print(ds1)

    assert ds1.shape[0] == 10
    assert ds2.shape[0] == 10
    assert ds3.shape[0] == 10
