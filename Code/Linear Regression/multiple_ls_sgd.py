import numpy as np 
from base_class import BaseRegression


class LinearRegressionSGD(BaseRegression):

    def __init__(self,itr=1000,alpha=0.01):

        self.cfs = None 
        self.intercept = None
        self.itr = itr
        self.alpha = alpha
        self.cost_history = []  # Track cost for monitoring convergence

    def fit(self,X_train,y_train):

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

        n = X_train.shape[1]

        beta = np.zeros(n)

        for i in range(self.itr):

            # Simply adding a batch size can covert this into a mini batch gradient decent 
            idx = np.random.randint(0,len(X_train))

            X_train_batch = X_train[idx]
            y_train_batch = y_train[idx]

            y_pred_batch = np.dot(X_train_batch,beta)

            error = y_train_batch - y_pred_batch

            grad = - np.dot(X_train_batch.T, error) #/ len(y_train_batch)

            beta -= self.alpha*grad

            y_pred_full = np.dot(X_train, beta)
            error_full = y_train - y_pred_full

            # Calculate the Mean Squared Error (MSE) over the entire training set
            cost = np.mean(error_full ** 2)
            self.cost_history.append(cost)

            print(f'Iteration {i+1}: MSE = {cost:.6f}')

        # Store final parameters
        self.intercept = beta[0]
        self.cfs = beta[1:]
    
    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float64)
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        y_pred = np.dot(X_test, [self.intercept] + self.cfs.tolist())
        return y_pred