import json
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from meta_act.metalearn import MetaLearner

METADB_PATH = "./test/data/metadb_test.csv"


def test_metaleaner_predict():
    metadb = pd.read_csv(METADB_PATH)
    learner = MetaLearner(RandomForestRegressor)

    print(metadb)

    X = metadb.iloc[:, :-9]
    y = metadb.iloc[:, -1]

    learner.fit(X, y)

    results = learner.test(X, y)
    vals = learner.predict(X.iloc[1:5])

    print(results)
    print(vals)

    assert len(vals) == 4
    for val in vals:
        assert isinstance(val, float)
    assert "MSE" in results
    assert "MAE" in results
    assert "R^2-Test" in results


def test_save_load_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = f"{tmpdir}/learner.joblib"
        metadt_path = f"{tmpdir}/learner_metadata.json"
        metadb = pd.read_csv(METADB_PATH)
        learner = MetaLearner(RandomForestRegressor)

        print(metadb)

        X = metadb.iloc[:, :-9]
        y = metadb.iloc[:, -1]

        learner.fit(X, y, oversample=False)
        learner.save_model(model_path)

        new_learner = MetaLearner(model_path)
        results = new_learner.test(X, y)
        vals = new_learner.predict(X.iloc[1:5])

        print(results)
        print(vals)

        with open(metadt_path, "r") as f:
            metadata = json.load(f)

        assert Path(model_path).exists()
        assert "sklearn_version" in metadata
        assert metadata["train_samples"] == X.shape[0]
        assert len(vals) == 4
        for val in vals:
            assert isinstance(val, float)
        assert "MSE" in results
        assert "MAE" in results
        assert "R^2-Test" in results
