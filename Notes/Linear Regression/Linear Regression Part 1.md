
Linear regression is a supervised machine learning algorithm which deals with predicting a quantitative (dependent) variable given one or more predictors (independent variables) . 

Often described as a line fitting problem, idea is to fit a line which minimizes the distance between the line and the actual values of the quantitative (dependent) variable.

On the basis of number of predictors there are two kinds of linear regression:
1. Simple Linear Regression: only one predictor. 
2. Multiple Linear Regression: multiple predictors. 

### Simple Linear Regression: 

In this variant only one predictor is present, and the relationship is represented by the equation 

$$Y = \beta_{0} + \beta_1X + \epsilon $$
Where $Y$ is the dependent variable,
	   $X$ is the independent variable
	   $\beta_0$ and $\beta_1$ are learnable parameters
	   and $\epsilon$ is the error

### Multiple Linear Regression 
In this method only multiple predictors are present, and the relationship is represented by the equation 

$$Y = \beta_0+\beta_1X+\dots+\beta_nX_n + \epsilon$$
Where $Y$ is the dependent variable,
	   $X_i$'s is the independent variable
	   $\beta_i$'s are learnable parameters
	   $n$ are the number of predictors 
	   and $\epsilon$ is the error
	   
## Assumptions

Behind the success of any ML model assumptions play an important role, as we are trying to map a function which will help us to establish a relationship between the predictors and the target variable. Below is a attempt to understand these assumptions, how they affect the model building process 

#### 1. Linearity

Its is assumed that the predictors and outcome is linear in parameters. The intuition behind this is we are trying to fit a line (hyperplane in higher dimension) which can best explain the response variable.

Note: If the relationship between the variables are non-linear then results of the model will be biased in nature under this assumption.

In practice , to verify this: 
- It is a good practice to calculate correlation between the variables, as it also makes the same assumption of linear relationship between the variables 
- One can also plot a scatter plot to check if the relationship holds or not. 
#### 2. Independence of Observations

Each observation $(X_i,Y_i)$  is independent of other.   

#### 3. No Perfect Multicollinearity

The features must be independent of each other if the features are dependent or have a high collinearity, it will affect the matrix inversion. 

Also while working with highly correlated features, it is not possible to explain which feature has how much effect on the outcome variable.

- VIF score is used to verify this.
#### 4. Homoscedasticity 

The variance of the error terms is constant across all values of $X$:

$$Var(\epsilon_i|X_i) = \sigma^2$$
- can be check via plotting a residual plot wrt to response variable.

## How to estimate these parameters ?

Finding the optimal value of these parameters so that they best fit (estimate) the data points is a challenge. 

Usually, ordinary least square method is used for the estimation of parameters (sk-learn implementation).  But for experimentation and understanding of optimization techniques, three methods are discussed 

1. OLS - Ordinary Least Squares.
2. [[Gradient Decent]] Methods.
	- Batch Gradient Decent 
	- Stochastic Gradient Decent 
	- Mini Batch Gradient Decent 
3. Evolutionary Algorithms

It will be a good exercise to use evolutionary algorithms in trying to find the best parameters for linear regression. 


## Metrics for Evaluation 

When building regression models, we want to know **how well the model predicts continuous outcomes**. 

1. Mean Absolute Error
2. Mean Square Error
3. Root Mean Square Error
4. $r^{2}$ Score
5. Adjusted $r^2$ Score

#### 1. Mean Absolute Error 
 
 The Mean Absolute Error is the average of the absolute differences between the predicted and actual values.
 $$MAE = \frac{1}{n} \sum_{i}^{n}|{Y_{i}-\hat{Y_{i}}}|$$
Advantages: 
- The error is in same unit as the response variable y.
- More robust to outliers than other metrics.

Disadvantages:
- Not differentiable at zero, not recommended for gradient based algorithms. 

```
def mean_abs_error(y_true,y_pred):
	
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	
	mae = np.mean(np.abs(y_true-y_pred))
	
	return mae
```


#### 2. Mean Squared Error

The Mean Squared Error is the average of the squared differences between the predicted and actual values.

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

- **Advantage:**
    - **Differentiable**: The function is differentiable, making it a popular choice as a loss function in many machine learning algorithms.

- **Disadvantages:**
    - **Unit is squared**: The units of MSE are the square of the response variable's units, which makes interpretation less intuitive than MAE.
    - **Not robust to outliers**: Due to the squaring of errors, large differences between predicted and actual values have a disproportionately large impact on the final MSE score.

```
def mean_abs_error(y_true,y_pred):
	
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	
	mse = np.mean((y_true-y_pred)**2)
	
	return mse
```

#### 3. Root Mean Squared Error.

The square root of MSE. It can be thought of as the standard deviation of residuals (for unbiased models).
$$
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$
**Advantages:**

- Same unit as the original variable (preferred over MSE).
- Interpretable as the standard deviation of prediction errors.

Disadvantages:

- Sensitive to outliers 
- Doesn’t decompose linearly; combining or averaging RMSEs across datasets is not always straightforward.

```
def mean_abs_error(y_true,y_pred):
	 
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	
	mse = np.mean((y_true-y_pred)**2)
	
	return mse
```


#### 4. $R^2$ Score (Coefficient of Determination)

It tell how much variation in the data can be explain by the model. 

Take, total variance of the dependent variable is 

$$TSS = \sum_{i}^{n}(Y_{i}-\bar{Y_{i}})^2$$
which is called total sum of squares, and the variance from predictions is quantified by residuals more exactly the the total sum of residuals. 

$$ RSS = \sum_{i}^{n}(Y_{i}-\hat{Y_{i}})^2$$
Now, mathematically we can define $R^2$ score as

$$R^2 = 1- \frac{RSS}{TSS} = 1-\frac{\sum_{i}^{n}(Y_{i}-\hat{Y_{i}})^2}{\sum_{i}^{n}(Y_{i}-\bar{Y_{i}})^2}$$
*Limitation of $R^2$* : Usually with the increase in the number of predictors $R^2$ score increase, leading to higher computational cost.

```
def r2_score(y_true,y_pred):
	
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	
	rss = np.sum((y_true-y_pred)**2)
	tss = np.sum((y_true-np.mean(y_true))**2)
	
	r2 = 1-(rss/tss)
	
	return r2
```
#### 5. Adjusted $R^2$ Score 

The adjusted $R^{2}$ score improves upon the regular $R^2$ by adjusting for the number of predictors (independent variables) used in the model.  
While $R^2$ always increases (or stays the same) when new variables are added — even if they are irrelevant the Adjusted $R^{2}$ penalizes unnecessary predictors that don’t actually improve model performance.

$$
\bar{R}^2 = 1 - (1 - R^2) \cdot \frac{n - 1}{n - k - 1}
$$

Where:
- $n$: number of observations.
- $k$: number of predictors.
- $R^{2}$: coefficient of determination.

```
def adjusted_r_squared(y_true, y_pred, X_train):
	
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	
    n = len(y_true)        
    k = X_train.shape[1]   
    
    rss = np.sum((y_true-y_pred)**2)
	tss = np.sum((y_true-np.mean(y_true))**2)
	
	r2 = 1-(rss/tss)
    
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - k - 1))
    return adj_r2
```




