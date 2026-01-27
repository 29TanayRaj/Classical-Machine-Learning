import numpy as np

class SVM:
    def __init__(self, lr=0.001, C=1.0, epochs=1000):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        # y must be in {-1, +1}
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1
                if condition:
                    # Only regularization term
                    self.w -= self.lr * self.w
                else:
                    # Hinge loss gradient
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b -= self.lr * (-self.C * y[i])

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)