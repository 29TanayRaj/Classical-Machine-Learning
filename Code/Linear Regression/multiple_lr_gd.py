import numpy as np 
from base_class import BaseRegression


class LinearRegressionGD(BaseRegression):

    def __init__(self,itr=1000,alpha=0.01):

        self.cfs = None 
        self.intercept = None
        self.itr = itr
        self.alpha = alpha
        self.cost_history = []  # Track cost for monitoring convergence


    def fit(self, X_train, y_train):
        X_train = np.array(X_train,dtype=np.float64)
        y_train = np.array(y_train,dtype=np.float64).flatten()

        # Add bias column (intercept term)
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        
        # Number of parameters in the linear model 
        n = X_train.shape[1]

        # Initialize the parameters 
        beta = np.zeros(n)

        # Optimization loop 
        for i in range(self.itr):
            # Forward pass
            y_pred = np.dot(X_train, beta)
            
            # Calculate error
            error = y_train - y_pred
            
            # Calculate gradient
            grad = -2 * np.dot(X_train.T, error) / len(y_train)
            
            # Update parameters
            beta -= self.alpha * grad
            
            # Store cost for monitoring (optional)
            cost = np.mean(error**2)
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
