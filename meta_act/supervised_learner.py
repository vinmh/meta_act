class SupervisedLearner:
    def __init__(self, model):
        self.model = model
        self.accuracy = 0
        self.hits = 0
        self.miss = 0
        self.samples_seen = 0

    def prequential_eval(self, X, y, target_values):
        # Evaluation
        y_hat = self.model.predict(X)
        pred_compare = y == y_hat
        new_hits = len([x for x in pred_compare if x])
        new_miss = len([x for x in pred_compare if not x])

        self.hits += new_hits
        self.miss += new_miss
        self.accuracy = self.hits / (
            self.hits + self.miss if (self.hits + self.miss != 0) else 1
        )

        # Train
        self.model.partial_fit(X, y, classes=target_values)

        return pred_compare

    def next_data(self, X, y, target_values):
        hit = self.prequential_eval(X, y, target_values)
        self.samples_seen += 1
        return self.hits, self.miss, self.accuracy, hit
