import math

import tsfel
from scipy.stats import entropy

from meta_act.windows import get_window_features


class ActiveLearner:
    def __init__(
            self, z_val, stream, model, budget=None, budget_window=None,
            grace_period=None, store_history=False
    ):
        self.z_val = z_val
        self.model = model
        self.grace_period = (getattr(model, "grace_period", 200) if
                             grace_period is None else grace_period)
        self.stream = stream
        self.budget = budget
        self.budget_window = (budget_window if budget_window is not None
                              else 1000)
        self.budget_counter = 0
        self.accuracy = 0
        self.hits = 0
        self.miss = 0
        self.queries = 0
        self.samples_seen = 0
        self.history = None if not store_history else []

        X, y = self.stream.next_sample(self.grace_period)
        self._prequential_eval(X, y)

    def _prequential_eval(self, X, y, query=True):
        # Evaluation
        y_hat = self.model.predict(X)
        pred_compare = y == y_hat
        new_hits = len(list(filter(lambda x: x, pred_compare)))
        new_miss = len(list(filter(lambda x: not x, pred_compare)))

        self.hits += new_hits
        self.miss += new_miss
        self.accuracy = self.hits / (
            self.hits + self.miss if (self.hits + self.miss != 0) else 1
        )

        # Train
        if query:
            self.queries += 1
            self.model.partial_fit(X, y, classes=self.stream.target_values)

    def next_data(self):
        if self.stream.has_more_samples():
            X, y = self.stream.next_sample()
            probs = self.model.predict_proba(X)
            entropy_val = entropy(probs[0], base=2)

            if self.budget is not None:
                if (
                        self.budget_counter <= self.budget
                        and self.budget_counter < self.budget_window
                ):
                    query = False
                    self.budget_counter += 1
                elif self.budget_counter >= self.budget_window:
                    query = False
                    self.budget_counter = 0
                else:
                    query = (
                            entropy_val >= self.z_val
                            or entropy_val == 0
                            or math.isnan(entropy_val)
                    )
            else:
                query = (
                        entropy_val >= self.z_val
                        or entropy_val == 0
                        or math.isnan(entropy_val)
                )

            self._prequential_eval(X, y, query)
            if self.history is not None:
                self.history.append((X, y))
            self.samples_seen += 1
            return self.hits, self.miss, self.accuracy, query
        else:
            return self.hits, self.miss, self.accuracy, None

    def get_last_window(self, mfe_features=None, tsfel_config=None,
                        features_summaries=None):
        if features_summaries is None:
            features_summaries = ["max", "min", "mean", "var"]
        if mfe_features is None:
            mfe_features = ["nr_class", "attr_ent", "kurtosis", "skewness"]
        if tsfel_config is None:
            tsfel_config = tsfel.get_features_by_domain()
        if self.history is not None:
            X = [sample for x in self.history for sample in x[0]]
            features = get_window_features(X, mfe_features, tsfel_config,
                                           features_summaries)
            self.history = []
            return features
        else:
            return None
