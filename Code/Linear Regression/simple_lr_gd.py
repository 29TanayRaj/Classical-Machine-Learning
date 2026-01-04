import numpy as np
from base_class import BaseRegression

# Simple Linear Regression Gradient Decent implementation.
class SimpleLinearRegressionGD(BaseRegression):

    def __init__(self, itr=50, alpha=0.01):
        super().__init__()
        self.slope = None
        self.intercept = None
        self.alpha = alpha
        self.itr = itr

    def fit(self, X_train, y_train):
        X_train = np.array(X_train).flatten()
        y_train = np.array(y_train).flatten()

        beta_0 = 0.0
        beta_1 = 0.0

        for i in range(self.itr):
            y_pred = beta_0 + beta_1 * X_train

            beta_0_grad = -2 * np.mean(y_train - y_pred)
            beta_1_grad = -2 * np.mean((y_train - y_pred) * X_train)

            beta_0 -= self.alpha * beta_0_grad
            beta_1 -= self.alpha * beta_1_grad

            print(f'Loss {i+1} :', np.mean((y_train - y_pred) ** 2))

        self.intercept = beta_0
        self.slope = beta_1
        self.weights = [beta_0, beta_1]

    def predict(self, X_test):
        X_test = np.array(X_test).flatten()
        return self.intercept + self.slope * X_test
