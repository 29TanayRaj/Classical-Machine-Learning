import numpy as np


# This is a peceptron used for binary classification
class Perceptron:

    def __init__(self,learning_rate,iterations,store_history=False,thershold = 0):
        
        """
        A simple Perceptron for binary classification.

        This model learns a linear decision boundary using a basic update rule.
        On each iteration, it randomly picks a data point and adjusts the weights
        if the prediction is wrong.

        P.s Edge cases are not handled in the algorithem

        Parameters
        ----------
        learning_rate : float
            How big each weight update step should be.

        iterations : int
            How many updates to perform during training.

        store_history : bool, optional
            If True, keeps track of how the weights change over time.

        thershold : float, optional
            The cutoff value used to decide between class 0 and 1.

        Attributes
        ----------
        w : ndarray
            The learned weights (including bias).

        weights_history : list or None
            Stores weight updates if enabled, otherwise None.
        """

        self.lr =  learning_rate
        self.itr = iterations
        self.w = None
        self.store_history = store_history
        self.ths = thershold  
        self.weights_history = None

    def fit(self,X,y):

        if self.store_history:
            self.weights_history = []

        X_cal = np.hstack((np.ones((X.shape[0], 1)), X))
        y_cal = np.array(y)

        # random initialization of w
        self.w = np.random.randn(X_cal.shape[1])

        for _ in range(self.itr):
            # random point selection
            i = np.random.randint(0, X_cal.shape[0])
            
            # calculating y pred (step function)
            z = np.dot(X_cal[i], self.w)

            y_pred = 1 if z >= self.ths else 0
            
            # updating the weights (perceptron update rule)
            self.w = self.w + self.lr * (y_cal[i] - y_pred) * X_cal[i]

            if self.store_history:
                self.weights_history.append(self.w.copy())

        return self
    
    def predict(self, x_test):

        if self.w is None:
            raise ValueError("Model not trained, call the fit method first")

        x_test = np.atleast_2d(x_test)

        x_test_cal = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

        y_raw = np.dot(x_test_cal, self.w)

        y_pred = (y_raw >= self.ths).astype(int)

        return y_pred if len(y_pred) > 1 else y_pred[0]