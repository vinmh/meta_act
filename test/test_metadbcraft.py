import json

import numpy as np
import pandas as pd

from meta_act.ds_gen import DatasetGeneratorPackage, DatasetGenerator
from meta_act.metadb_craft import (
    MetaDBCrafter, WindowResults, ZValGenerator, WindowGenerator
)


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


def test_window_get_stream(datadir):
    z_val_gen = ZValGenerator(4)
    wind_gen = WindowGenerator(z_val_gen, cache_dir=datadir / "test_cache")

    generator1 = DatasetGenerator(
        "HyperplaneGenerator", {"n_features": [5]}
    )
    package = DatasetGeneratorPackage([generator1], 10)
    cnt = 0
    for dat, name in wind_gen.get_stream(package):
        print(name)
        print(dat)
        if cnt == 0:
            assert name == "elecNormNew.npy"
            assert dat.shape == (45312, 8)
        else:
            assert name != "elecNormNew.npy"
            assert dat.shape[0] == 10
            break
        cnt += 1


def test_generate_windows(datadir):
    cache_dir = datadir / "test_cache"
    z_val_gen = ZValGenerator(4)
    wind_gen = WindowGenerator(
        z_val_gen, cache_dir=cache_dir, target_windows=10, window_package_size=2
    )

    generator1 = DatasetGenerator(
        "HyperplaneGenerator", {"n_features": [5, 10, 15]}
    )
    generator2 = DatasetGenerator(
        "LEDGeneratorDrift", {"noise_percentage": [0.0, 0.1, 0.5]}
    )
    package = DatasetGeneratorPackage([generator1, generator2], 1000)

    file_list = wind_gen.generate_windows(package)
    metadata_path = cache_dir / "windows" / wind_gen.WINDOW_METADATA_FILENAME
    assert metadata_path.exists()
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert len(metadata.keys()) >= 1
    assert len(file_list) == 5
    for file_name in file_list:
        filepath = cache_dir / "windows" / file_name
        assert filepath.exists()
        file_data = np.load(filepath)
        assert len(file_data) == 2


def test_parse_window_file(datadir):
    cache_dir = datadir / "test_cache/windows"
    z_val_gen = ZValGenerator(4)
    crafter = MetaDBCrafter(
        z_val_gen, window_cache_dir=cache_dir,
        mfe_features=["min", "mean", "max"], feature_summaries=["mean"]
    )
    filename = "0__10__elecNormNew.npy.npz"
    with open(cache_dir / "window_metadata.json") as f:
        metadata = json.load(f)

    samples = crafter.parse_window_file(filename, metadata)
    print(samples)

    assert samples.shape[0] == 10
    assert samples.shape[1] == 8


def test_create_metadb(datadir):
    cache_dir = datadir / "test_cache/windows"
    z_val_gen = ZValGenerator(4)
    crafter = MetaDBCrafter(
        z_val_gen, window_cache_dir=cache_dir,
        mfe_features=["min", "mean", "max"], feature_summaries=["mean"],
        output_path=datadir / "out.csv", threads_number=2
    )
    file_list = [
        "0__10__elecNormNew.npy.npz", "1__10__elecNormNew.npy.npz",
        "2__10__elecNormNew.npy.npz", "3__10__elecNormNew.npy.npz",
        "4__10__elecNormNew.npy.npz"
    ]

    metadb = crafter.create_metadb(file_list)

    assert metadb.shape[0] == 50
    assert (datadir / "out.csv").exists()
    assert pd.read_csv(datadir / "out.csv").shape == (50, 8)
