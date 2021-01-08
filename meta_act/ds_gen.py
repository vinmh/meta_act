import logging
import random
from pathlib import Path
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from skmultiflow import data

from meta_act.util import dict_str_creator


def generate_datasets(generator_name: str, outpath: str, max_samples=100000,
                      **kwargs):
    key1 = list(kwargs.keys())[0]
    val1 = kwargs.pop(key1)
    if not isinstance(val1, list):
        val1 = [val1]
    combinations = [{key1: v1} for v1 in val1]
    for k, v in kwargs.items():
        if not isinstance(v, list):
            v = [v]
        combinations = [dict(c, **{k: v1}) for c in combinations for v1 in v]

    file_prefix = generator_name.replace("Generator", "")
    genclass = getattr(data, generator_name, None)

    if genclass is None:
        raise ValueError(f"{generator_name} not found!")

    for combination in combinations:
        generator = genclass(**combination)
        X = []
        y = []
        while len(X) < max_samples and generator.has_more_samples():
            X1, y1 = generator.next_sample()
            X.append(X1)
            y.append(y1)

        X = np.concatenate(X)
        y = np.concatenate(y)

        df = pd.DataFrame(np.hstack((X, np.array([y], dtype=int).T)))

        filename = (f"{file_prefix}_"
                    f"{'_'.join([f'{k}V{v}' for k, v in combination.items()])}"
                    f".csv")
        outfile = Path(outpath) / filename

        logging.info(f"Writing {filename} dataset...")
        df.to_csv(outfile, index=False)


def dataset_generator(generators: List[Tuple[str, Dict[str, List[Any]]]],
                      max_samples=100000, max_datasets=None):
    gen_classes = []
    for gen in generators:
        genclass = getattr(data, gen[0], None)
        if genclass is None:
            raise ValueError(f"{gen[0]} not found!")
        gen_classes.append(genclass)

    gen_n = 0
    while gen_n < max_datasets if max_datasets is not None else 1:
        selected_gen_n = gen_n % len(generators)
        genclass = gen_classes[selected_gen_n]
        genmetadata = generators[selected_gen_n]

        hyperparams = {k: random.choice(v) for k, v in genmetadata[1].items()}
        logging.info(f"{gen_n}: Generating dataset from {genmetadata[0]}")
        logging.debug(f"Hyperparameters: {dict_str_creator(hyperparams)}")

        generator = genclass(**hyperparams)
        X = []
        y = []
        while len(X) < max_samples and generator.has_more_samples():
            X1, y1 = generator.next_sample()
            X.append(X1)
            y.append(y1)

        X = np.concatenate(X)
        y = np.concatenate(y)

        file_prefix = genmetadata[0].replace("Generator", "")
        filename = (f"{file_prefix}_"
                    f"{'_'.join([f'{k}V{v}' for k, v in hyperparams.items()])}")

        yield np.hstack((X, np.array([y], dtype=int).T)), filename
