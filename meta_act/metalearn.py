import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target


class MetaLearner:
    def __init__(self, learner, *learner_args, **learner_kwargs):
        if isinstance(learner, str):
            self.model = joblib.load(learner)
            filepath = Path(learner)
            metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
            scaler_path = filepath.parent / f"{filepath.stem}_scaler.joblib"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"sklearn_version": sklearn.__version__,
                                 "train_samples": 0}
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None
            self.trained = True
        else:
            self.model = learner(*learner_args, **learner_kwargs)
            self.metadata = {"sklearn_version": sklearn.__version__,
                             "train_samples": 0}
            self.trained = False
            self.scaler = None

    def _scale(self, X):
        if self.scaler is None:
            self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self.scaler.transform(X)

    def _eliminate_minority(self, X, y, threshold):
        tmp_dst = X.copy(deep=True)
        tmp_dst["Y"] = y

        minor = [val for val, _ in
                 filter(lambda x: x[1] < threshold,
                        zip(*np.unique(tmp_dst.Y, return_counts=True)))]
        dst_no_minor = tmp_dst.loc[~tmp_dst["Y"].isin(minor)]

        return dst_no_minor.iloc[:, :-1], dst_no_minor.iloc[:, -1]

    def fit(self, X, y, oversample=True, scale=True, test_data=None,
            eliminate_minority=True, minority_threshold=50):
        if eliminate_minority:
            X, y = self._eliminate_minority(X, y, minority_threshold)
            if test_data is not None and isinstance(test_data, tuple):
                test_data = self._eliminate_minority(test_data[0], test_data[1],
                                                     minority_threshold)

        if scale:
            X = self._scale(X)
            if test_data is not None and isinstance(test_data, tuple):
                test_data = (self._scale(test_data[0]), test_data[1],)

        if oversample:
            continuous = False
            if type_of_target(y) == "continuous":
                original_dtype = y.dtype
                y = y.astype('str')
                continuous = True
            smote = SMOTE()
            X, y = smote.fit_resample(X, y)
            if continuous:
                y = y.astype(original_dtype)

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

    def test(self, X, y, scale=True, eliminate_minority=True,
             minority_threshold=50):
        if not self.trained:
            raise ValueError("Model not trained!")
        if eliminate_minority:
            X, y = self._eliminate_minority(X, y, minority_threshold)
        if scale:
            X = self._scale(X)
        results = {"R^2-Test": self.model.score(X, y)}
        test_predict = self.model.predict(X)
        results["MSE"] = mean_squared_error(y, test_predict)
        results["MAE"] = mean_absolute_error(y, test_predict)

        self.metadata = dict(self.metadata, **results)
        return results

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model not trained!")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def save_model(self, filepath):
        if not self.trained:
            raise ValueError("Model not trained!")
        filepath = Path(filepath)
        with open(filepath.parent / f"{filepath.stem}_metadata.json", "w") as \
                mtdt:
            json.dump(self.metadata, mtdt)
        joblib.dump(self.model, filepath.as_posix())
        if self.scaler is not None:
            joblib.dump(self.scaler,
                        filepath.parent / f"{filepath.stem}_scaler.joblib")
