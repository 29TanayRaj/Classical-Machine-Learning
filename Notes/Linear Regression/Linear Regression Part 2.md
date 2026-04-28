Finding the best parameter in the model is often a challenge, for linaer regression there are mutiple ways to finding these parameters.

### 1. Closed form solution

The idea is to come up with a closed form solution for parameters, first we will do this exercise for simple linear regression.

The derivation starts by idea of minimizing the error

$$e^2 = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

where,

$$\hat{y}_i = \beta_0 + \beta_1 x_i$$

Substituting the values in the equation above we get

$$L = e^2 = \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2$$

**Partial derivative w.r.t. $\beta_0$:**

$$\frac{\partial L}{\partial \beta_0} = \sum_{i=1}^{n} 2(y_i - \beta_0 - \beta_1 x_i)(-1) = 0$$

$$\Rightarrow \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i) = 0$$

$$\Rightarrow \sum_{i=1}^{n} y_i - n\beta_0 - \beta_1 \sum_{i=1}^{n} x_i = 0$$

$$\boxed{\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}}$$


**Partial derivative w.r.t. $\beta_1$:**

$$\frac{\partial L}{\partial \beta_1} = \sum_{i=1}^{n} 2(y_i - \beta_0 - \beta_1 x_i)(-x_i) = 0$$

$$\Rightarrow \sum_{i=1}^{n} x_i (y_i - \beta_0 - \beta_1 x_i) = 0$$

$$\Rightarrow \sum_{i=1}^{n} x_i y_i - \beta_0 \sum_{i=1}^{n} x_i - \beta_1 \sum_{i=1}^{n} x_i^2 = 0$$


Substituting $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$:

$$\sum_{i=1}^{n} x_i y_i - (\bar{y} - \hat{\beta}_1 \bar{x}) \sum_{i=1}^{n} x_i - \hat{\beta}_1 \sum_{i=1}^{n} x_i^2 = 0$$

$$\sum_{i=1}^{n} x_i y_i - n\bar{x}\bar{y} + \hat{\beta}_1 n\bar{x}^2 - \hat{\beta}_1 \sum_{i=1}^{n} x_i^2 = 0$$

$$\hat{\beta}_1 \left(\sum_{i=1}^{n} x_i^2 - n\bar{x}^2 \right) = \sum_{i=1}^{n} x_i y_i - n\bar{x}\bar{y}$$

Recognising the sums-of-squares notation:

$$S_{xx} = \sum_{i=1}^{n}(x_i - \bar{x})^2 = \sum_{i=1}^{n} x_i^2 - n\bar{x}^2$$

$$S_{xy} = \sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}) = \sum_{i=1}^{n} x_i y_i - n\bar{x}\bar{y}$$


**OLS Estimators:**

$$\boxed{\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}} 
= \frac{\displaystyle \sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
{\displaystyle \sum_{i=1}^{n}(x_i - \bar{x})^2}}$$

$$\boxed{\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}}$$


## Multiple Linear Regression — OLS Derivation in Matrix Form


### The model

With $n$ observations and $p$ predictors, the multiple linear regression model is:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i, \quad i = 1, 2, \ldots, n$$


### Matrix notation


Stack all observations into vectors and matrices:

$$\mathbf{y} =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}_{n \times 1}, \quad
\mathbf{X} =
\begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}_{n \times (p+1)}
$$

$$\boldsymbol{\beta} =
\begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_p
\end{bmatrix}_{(p+1) \times 1}, \quad
\boldsymbol{\varepsilon} =
\begin{bmatrix}
\varepsilon_1 \\
\varepsilon_2 \\
\vdots \\
\varepsilon_n
\end{bmatrix}_{n \times 1}
$$

So the model compactly becomes:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$


### The loss function

We minimise the sum of squared residuals:

$$L(\boldsymbol{\beta}) = \sum_{i=1}^{n} \varepsilon_i^2 
= \boldsymbol{\varepsilon}^\top \boldsymbol{\varepsilon}
= (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Expanding the product:

$$L(\boldsymbol{\beta}) =
\mathbf{y}^\top \mathbf{y}
- \mathbf{y}^\top \mathbf{X}\boldsymbol{\beta}
- \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{y}
+ \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{X}\boldsymbol{\beta}$$

Since $\mathbf{y}^\top \mathbf{X}\boldsymbol{\beta}$ is a scalar, it equals its own transpose:

$$
\mathbf{y}^\top \mathbf{X}\boldsymbol{\beta}
= \left(\mathbf{y}^\top \mathbf{X}\boldsymbol{\beta}\right)^\top
= \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{y}
$$

Therefore:

$$
L(\boldsymbol{\beta}) =
\mathbf{y}^\top \mathbf{y}
- 2\boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{y}
+ \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{X}\boldsymbol{\beta}
$$


### Taking the derivative and setting to zero

Differentiate $L$ with respect to $\boldsymbol{\beta}$ using standard matrix calculus identities:

$$
\frac{\partial}{\partial \boldsymbol{\beta}}
\left(\boldsymbol{\beta}^\top \mathbf{a}\right) = \mathbf{a}, \quad
\frac{\partial}{\partial \boldsymbol{\beta}}
\left(\boldsymbol{\beta}^\top \mathbf{A} \boldsymbol{\beta}\right)
= 2\mathbf{A}\boldsymbol{\beta}
\quad \text{(when $\mathbf{A}$ is symmetric)}
$$

Applying these:

$$
\frac{\partial L}{\partial \boldsymbol{\beta}} =
-2\mathbf{X}^\top \mathbf{y}
+ 2\mathbf{X}^\top \mathbf{X}\boldsymbol{\beta}
= \mathbf{0}
$$


### The normal equations

Rearranging gives the **normal equations**:

$$
\mathbf{X}^\top \mathbf{X}\,\boldsymbol{\beta}
= \mathbf{X}^\top \mathbf{y}
$$

This is a system of $(p+1)$ linear equations in $(p+1)$ unknowns.

### Solving for $\hat{\boldsymbol{\beta}}$

Provided $\mathbf{X}^\top \mathbf{X}$ is invertible (i.e., $\mathbf{X}$ has full column rank — no perfect multicollinearity), we pre-multiply both sides by $(\mathbf{X}^\top \mathbf{X})^{-1}$:

$$
(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{X}\,\hat{\boldsymbol{\beta}}
= (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
$$

$$
\mathbf{I}\,\hat{\boldsymbol{\beta}}
= (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
$$

$$
\boxed{
\hat{\boldsymbol{\beta}}
= (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
}
$$


### Confirming it is a minimum (second-order condition)

The Hessian of $L$ with respect to $\boldsymbol{\beta}$ is:

$$
\mathbf{H} = \frac{\partial^2 L}{\partial \boldsymbol{\beta}\,\partial \boldsymbol{\beta}^\top}
= 2\mathbf{X}^\top \mathbf{X}
$$

For any non-zero vector $\mathbf{v} \in \mathbb{R}^{p+1}$:

$$
\mathbf{v}^\top \mathbf{H} \mathbf{v}
= 2\mathbf{v}^\top \mathbf{X}^\top \mathbf{X} \mathbf{v}
= 2\|\mathbf{X}\mathbf{v}\|^2 \ge 0
$$

Since $\mathbf{X}$ has full column rank, $\|\mathbf{X}\mathbf{v}\|^2 > 0$ for all $\mathbf{v} \ne \mathbf{0}$, so $\mathbf{H}$ is **positive definite** and the critical point is indeed a global minimum.


### Fitted values and residuals

The vector of fitted values is:

$$
\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}
= \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
= \mathbf{H}\mathbf{y}
$$

where

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top
$$

is the **hat matrix** (projection matrix onto the column space of $\mathbf{X}$).

The residual vector is:

$$
\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}}
= (\mathbf{I} - \mathbf{H})\mathbf{y}
$$

Note that $\mathbf{H}$ and $(\mathbf{I} - \mathbf{H})$ are both symmetric and idempotent:

$$
\mathbf{H}^2 = \mathbf{H}, \quad
(\mathbf{I} - \mathbf{H})^2 = \mathbf{I} - \mathbf{H}
$$

### Properties of the OLS estimator

Under the Gauss–Markov assumptions 
($\mathbb{E}[\boldsymbol{\varepsilon}] = \mathbf{0}$, 
$\mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}$):

#### **Unbiasedness:**

$$
\mathbb{E}[\hat{\boldsymbol{\beta}}]
= \mathbb{E}\left[(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}\right]
= (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbb{E}[\mathbf{y}]
$$

$$
= (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top (\mathbf{X}\boldsymbol{\beta})
= \boldsymbol{\beta}
$$

#### **Variance–covariance matrix:**

$$
\mathrm{Var}(\hat{\boldsymbol{\beta}})
= \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}
$$

#### **BLUE property (Gauss–Markov theorem):**

$\hat{\boldsymbol{\beta}}$ is the **Best Linear Unbiased Estimator (BLUE)**, it has the smallest variance among all linear unbiased estimators.

#### **Unbiased estimator of $\sigma^2$:**

$$
\hat{\sigma}^2
= \frac{\hat{\boldsymbol{\varepsilon}}^\top \hat{\boldsymbol{\varepsilon}}}{n - p - 1}
= \frac{\mathrm{RSS}}{n - p - 1}
$$

### Key results

| Quantity | Expression |
|----------|-----------|
| OLS estimator | $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$ |
| Fitted values | $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ |
| Hat matrix | $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top$ |
| Residuals | $\hat{\boldsymbol{\varepsilon}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$ |
| Variance of $\hat{\boldsymbol{\beta}}$ | $\sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}$ |
| Estimate of $\sigma^2$ | $\hat{\sigma}^2 = \frac{\mathrm{RSS}}{n - p - 1}$ |