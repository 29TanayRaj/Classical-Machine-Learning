import numpy as np 

class LogisticRegressionGD:
    '''
    Used for Binary Classification
    '''

    def __init__(self, alpha=0.05, itr=1000, thershold=0.5, beta=None):
        self.alpha = alpha 
        self.beta = beta 
        self.itr = itr
        self.thershold = thershold

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # add a one to the X_train to handle β₀
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]

        # randomly initialize parameters for the model sigmoid(βᵀX)
        self.beta = np.random.random(X_train.shape[1])

        # training
        for _ in range(self.itr):
            y_pred = self._sigmoid(X_train @ self.beta)
            grad = X_train.T @ (y_train - y_pred)
            self.beta = self.beta + self.alpha * grad

    def predict_prob(self, X_test):
        X_test = np.array(X_test)
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return self._sigmoid(X_test @ self.beta)
    
    def predict(self, X_test):
        probs = self.predict_prob(X_test)
        y_pred = np.array([1 if prob > self.thershold else 0 for prob in probs])
        return y_pred

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))
