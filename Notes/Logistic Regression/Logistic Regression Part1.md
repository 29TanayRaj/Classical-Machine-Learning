## Introduction to Logistic Regression 

Logistic regression is a supervised learning task in which we predict a binary target variable. This is a classification task.

$$ \sigma(z)= \frac{1}{1+\exp(-z)}$$

$$z = \beta_{0} + \beta_{1}x_{1}+\dots+\beta_{n}x_{n}$$

> The idea is to extend the linear regression into a probabilistic model, which squeeze the output between 0 and 1.

Logistic regression is a part of the generalized linear regression models. 

### Assumptions for logistic regression
- Logistic regression works on the assumption that the data points are linearly seperable. 

- All the observartions are independent os earch other. 

- No muticollinarlity between the predictors.

> Note: The core logistic regression algorithem can only do binary classification, but this algorithem can be extendened for mutiple classifications too via the use of one vs all, that will be discussed later  

### Motivation behind the whole problem

Logistic regression work by finding the best fit line which can divide the points into two parts, by the nature of algorithem this is only possible when the data points are linearly sepearble. 

we will now work on understanding how this algorithem works and the idea behind this. For this we start with the peceptron model.

## Perceptron Model

This model works fitting a line, which divides the space into a positive region and a negative region which helps us to classify the points. 

Lets say we have $n$ predictors $x_{1},x_{2},\dots x_{n}$, the we have a line 

$$\sum_{i=0}^{n} w_{i}x_{i} = 0$$

Which divides the plain into two regions which helps us to classify our datapoints, we use a step function to classify the data points

$$\hat{y} = f(z) =
\begin{cases}
0 & \text{if } z < 0 \\
1 & \text{if } z \ge 0
\end{cases} 
$$

$$\text{where, } z = \sum_{i=0}^{n} w_{i}x^{'}_{i}$$

#### How do we find this line?

The idea is to use missclassified points to push the points beyond the points. 

lets say we have a point $x^{1} = (x^{1}_{0},x^{1}_{1} \dots x^{1}_{n})$, which is supposed to be in the negative part of the plane (y=0), but is in positive part of the plane (y=1).

we would update the weights in such a way that the point is pushed toward the correct side of the decision boundary.


If $ x_i \in \mathcal{N} $ and $ \sum_{i=0}^{n} w_i x_i \ge 0 $, then

$$
w_{\text{new}} = w_{\text{old}} - \eta x_i
$$

If $ x_i \in \mathcal{P} $ and $ \sum_{i=0}^{n} w_i x_i < 0 $, then

$$
w_{\text{new}} = w_{\text{old}} + \eta x_i
$$

Combining the equations we get, for misclassified points. 

$$
w_{\text{new}} = w_{\text{old}} + \eta(y_{i}-\hat{y}_{i}) x_i
$$

Which is gives the perceptron learning algorithem.

```
Algorithm: Perceptron Training

Input:
    X → feature matrix of size (m × n)
    y → labels (0 or 1)
    epochs → number of iterations
    lr → learning rate

Step 1: Add bias term
    For each sample in X:
        prepend 1 to the feature vector

Step 2: Initialize weights
    w ← random vector of size (n + 1)

Step 3: Training loop
    For epoch = 1 to epochs:
        
        • Randomly select an index i from {0, 1, ..., m-1}
        
        • Compute linear combination:
            z ← dot(X[i], w)
        
        • Apply step function:
            If z ≥ 0:
                y_pred ← 1
            Else:
                y_pred ← 0
        
        • Update weights:
            w ← w + lr × (y[i] − y_pred) × X[i]

Step 4: Return learned weights w
```

Here is the equivalent python code for this: 

```python
import numpy as np

# adding one in the start of x (bias term)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# random initialization of w
w = np.random.randn(X.shape[1])

for _ in range(epochs):
    # random point selection
    i = np.random.randint(0, X.shape[0])
    
    # calculating y pred (step function)
    z = np.dot(X[i], w)
    y_pred = 1 if z >= 0 else 0
    
    # updating the weights (perceptron update rule)
    w = w + lr * (y[i] - y_pred) * X[i]
```

> Note: Though this gives a line that satifies our conditions for classification but there is no way of knowing if this is the best fit line.




