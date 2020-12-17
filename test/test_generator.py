import tempfile
from meta_act.ds_gen import generate_datasets
from pathlib import Path

def test_generate_datasets():
    generators = [("HyperplaneGenerator",
     {"n_features": [5, 10]}),
    ("LEDGeneratorDrift",
     {"noise_percentage": [0.0, 0.1]}),
    ("MIXEDGenerator",
     {"classification_function": [0, 1]}),
    ("RandomRBFGeneratorDrift",
     {"n_classes": [2, 3]}),
    ("RandomTreeGenerator",
     {"n_classes": [2, 3]}),
    ("SineGenerator",
     {"classification_function": [0, 1]}),
    ("STAGGERGenerator",
     {"classification_function": [0, 1]}),]

    with tempfile.TemporaryDirectory() as tmpdir:
        for generator in generators:
            generate_datasets(generator[0],
                              tmpdir,
                              max_samples=10,
                              **generator[1])

        dataset_files = list(Path(tmpdir).glob("*.csv"))
        assert len(dataset_files) == 2*len(generators)
