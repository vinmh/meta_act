import json
from pathlib import Path

import joblib
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MetaLearner:
    def __init__(self, learner, *learner_args, **learner_kwargs):
        if isinstance(learner, str):
            self.model = joblib.load(learner)
            filepath = Path(learner)
            metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"sklearn_version": sklearn.__version__,
                                 "train_samples": 0}
            self.trained = True
        else:
            self.model = learner(*learner_args, **learner_kwargs)
            self.metadata = {"sklearn_version": sklearn.__version__,
                             "train_samples": 0}
            self.trained = False

    def fit(self, X, y, oversample=True, test_data=None):
        if oversample:
            smote = SMOTE(random_state=20)
            X, y = smote.fit_resample(X, y)

        self.model.fit(X, y)
        results = {"R^2-Train": self.model.score(X, y)}

        if test_data is not None and isinstance(test_data, tuple):
            results["R^2-Test"] = self.model.score(test_data[0], test_data[1])
            test_predict = self.model.predict(test_data[0])
            results["MSE"] = mean_squared_error(test_data[1], test_predict)
            results["MAE"] = mean_absolute_error(test_data[1], test_predict)

        self.trained = True
        self.metadata["train_samples"] += len(X)
        self.metadata = dict(self.metadata, **results)
        return results

    def test(self, X, y):
        if not self.trained:
            raise ValueError("Model not trained!")
        results = {"R^2-Test": self.model.score(X, y)}
        test_predict = self.model.predict(X)
        results["MSE"] = mean_squared_error(y, test_predict)
        results["MAE"] = mean_absolute_error(y, test_predict)

        self.metadata = dict(self.metadata, **results)
        return results

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model not trained!")
        return self.model.predict(X)

    def save_model(self, filepath):
        if not self.trained:
            raise ValueError("Model not trained!")
        filepath = Path(filepath)
        with open(filepath.parent / f"{filepath.stem}_metadata.json", "w") as \
                mtdt:
            json.dump(self.metadata, mtdt)
        joblib.dump(self.model, filepath.as_posix())
