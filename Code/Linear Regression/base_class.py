import numpy as np

class BaseRegression:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X_train, y_train):
        raise NotImplementedError("fit method must be implemented in subclass")

    def predict(self, X_test):
        pass   
