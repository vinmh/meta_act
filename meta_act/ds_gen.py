import functools
import itertools
import logging
import random
from typing import List

import numpy as np
from skmultiflow import data

from meta_act.util import dict_str_creator


class DatasetGenerator:
    def __init__(self, name, kwargs_combinations, package=data):
        self.name = name.replace("Generator", "")
        self.gen_class = getattr(package, name, None)
        if self.gen_class is None:
            raise ValueError(f"{name} Generator not found!")
        kwargs_possibilities = []
        for k, v in kwargs_combinations.items():
            if not isinstance(v, list):
                v = [v]
            kwargs_possibilities.append([{k: v_e} for v_e in v])

        self.combinations = []
        for p in itertools.product(*kwargs_possibilities):
            self.combinations.append(
                functools.reduce(lambda x, y: {**x, **y}, p)
            )

    def random_combination(self):
        return self.combinations[random.randrange(0, len(self.combinations))]


class DatasetGeneratorPackage:
    def __init__(
            self,
            generators: List[DatasetGenerator],
            max_samples=100000
    ):
        self.generators = generators
        self.max_samples = max_samples

    def generate_dataset(self, gen_n):
        curr_gen = self.generators[gen_n % len(self.generators)]
        kwargs = curr_gen.random_combination()
        logging.debug(f"{gen_n}: Generating dataset from {curr_gen.name}")
        logging.debug(f"Hyper-parameters: {dict_str_creator(kwargs)}")

        generator = curr_gen.gen_class(**kwargs)
        X = []
        y = []
        while len(X) < self.max_samples and generator.has_more_samples():
            X1, y1 = generator.next_sample()
            X.append(X1)
            y.append(y1)

        X = np.concatenate(X)
        y = np.concatenate(y)

        filename = (
            f"{curr_gen.name}_"
            f"{'_'.join([f'{k}V{v}' for k, v in kwargs.items()])}"
        )

        return np.hstack((X, np.array([y], dtype=int).T)), filename
