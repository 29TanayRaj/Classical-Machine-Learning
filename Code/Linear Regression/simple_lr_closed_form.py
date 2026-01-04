import numpy as np
from base_class import BaseRegression

class SimpleLinearRegressionCF(BaseRegression):

    def __init__(self):
        super().__init__()
        self.weights = None
        self.intercept = None        
        self.slope = None             

    def fit(self,X_train,y_train):

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if np.var(X_train) == 0:
            raise ValueError("All X values are the same. Can't fit a line through constant X.")

        beta_1 = (np.mean(X_train*y_train) - np.mean(X_train)*np.mean(y_train))/(np.mean(X_train**2)-np.mean(X_train)**2)
        # beta_1 = np.sum((X_train - np.mean(X_train)))*np.sum((y_train - np.mean(y_train)))/(np.sum((X_train - np.mean(X_train))**2))

        beta_0 = np.mean(y_train) - beta_1*np.mean(X_train)

        self.weights = np.array([beta_0,beta_1])
        self.intercept = beta_0
        self.slope = beta_1

    def predict(self,X_test):

        X_test = np.array(X_test)

        y_pred = self.intercept + self.slope*X_test

        return y_pred
