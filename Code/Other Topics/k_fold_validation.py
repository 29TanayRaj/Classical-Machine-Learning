import numpy as np

class KFoldCV:
    def __init__(self, k: int, shuffle: bool = True, random_state: int | None = None):
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state
        self.scores_ = []

    def split(self, X):
        n = len(X)
        idx = np.arange(n)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(idx)

        # compute fold sizes
        fold_sizes = np.full(self.k, n // self.k)
        fold_sizes[: n % self.k] += 1

        # generate fold indices
        parts = []
        start = 0
        for size in fold_sizes:
            end = start + size
            parts.append(idx[start:end])
            start = end

        return parts

    def fit(self, model, X: list, y: list):
        X = np.array(X)
        y = np.array(y)

        parts = self.split(X)
        self.scores_ = []

        for i in range(self.k):
            test_idx = parts[i]
            train_idx = np.hstack([parts[j] for j in range(self.k) if j != i])

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # train
            model.fit(X_train, y_train)

            # predict
            y_pred = model.predict(X_test)

            # score
            score = model.score(y_test, y_pred)
            self.scores_.append(score)

        return self.scores_

    def summary(self):
        if not self.scores_:
            raise ValueError("Run fit() first.")

        mean_score = np.mean(self.scores_)
        print(f"The CV mean score is {mean_score}")
        print(f"Score at each fold: {self.scores_}")
        return mean_score
