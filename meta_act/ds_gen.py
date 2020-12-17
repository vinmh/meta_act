from skmultiflow import data
from pathlib import Path
import numpy as np
import pandas as pd

def generate_datasets(generator_name, outpath, max_samples=100000, **kwargs):
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

        df = pd.DataFrame(np.hstack((X,np.array([y]).T)))

        filename = f"{file_prefix}_{'_'.join([f'{k}V{v}' for k,v in combination.items()])}.csv"
        outfile = Path(outpath) / filename

        print(f"Writing {filename} dataset...")
        df.to_csv(outfile, index=False)
