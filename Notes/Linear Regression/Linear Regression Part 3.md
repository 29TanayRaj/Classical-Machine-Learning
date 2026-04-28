## Parameter Estimation via Gradient Descent

### Motivation

The closed-form OLS solution $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$ requires inverting an $(p+1) \times (p+1)$ matrix. This becomes computationally expensive — $\mathcal{O}(p^3)$ — when the number of features $p$ is large. Gradient descent sidesteps the matrix inversion entirely by iteratively nudging $\boldsymbol{\beta}$ in the direction that reduces the loss the fastest.

### The loss function

We use the Mean Squared Error (MSE) as our objective:

$$
L(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
$$

> Note: Some formulations use $\frac{1}{2n}$ to make the derivative cleaner. The factor does not affect the location of the minimum — only the effective learning rate.

In matrix form:

$$
L(\boldsymbol{\beta}) = \frac{1}{n}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$


### The gradient

Expand $L$:

$$
L(\boldsymbol{\beta}) = \frac{1}{n}\left(
\mathbf{y}^\top \mathbf{y}
- 2 \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{y}
+ \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta}
\right)
$$

Differentiate with respect to $\boldsymbol{\beta}$:

$$
\nabla_{\boldsymbol{\beta}} L
= \frac{\partial L}{\partial \boldsymbol{\beta}}
= \frac{1}{n}\left(
-2 \mathbf{X}^\top \mathbf{y}
+ 2 \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta}
\right)
$$

$$
\boxed{
\nabla_{\boldsymbol{\beta}} L
= \frac{-2}{n}\,\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
= \frac{-2}{n}\,\mathbf{X}^\top \hat{\boldsymbol{\varepsilon}}
}
$$

where $\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \mathbf{X}\boldsymbol{\beta}$ is the current residual vector.

**Intuition:** The gradient is the residuals $\hat{\boldsymbol{\varepsilon}}$ projected back through $\mathbf{X}^\top$. When the residuals are large, the gradient is large and the update step is bigger. At the optimum, $\mathbf{X}^\top \hat{\boldsymbol{\varepsilon}} = \mathbf{0}$ ,exactly the normal equations.


### Gradient descent variants

#### 1. Batch gradient descent (BGD)

Uses all $n$ observations to compute the gradient at each step.

$$
\boldsymbol{\beta}^{(t+1)} 
= \boldsymbol{\beta}^{(t)} 
- \alpha \, \nabla_{\boldsymbol{\beta}} L\!\left(\boldsymbol{\beta}^{(t)}\right)
$$

$$
\boldsymbol{\beta}^{(t+1)} 
= \boldsymbol{\beta}^{(t)} 
+ \frac{2\alpha}{n} \, \mathbf{X}^\top 
\left(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}^{(t)}\right)
$$

where $\alpha > 0$ is the **learning rate**.

- Stable, smooth convergence  
- Expensive per iteration when $n$ is large — requires a full pass over the data  

#### 2. Stochastic gradient descent (SGD)

Uses a single randomly sampled observation $(\mathbf{x}_i, y_i)$ per update:

$$
\nabla_{\boldsymbol{\beta}} L_i 
= -2 \, \mathbf{x}_i \left(y_i - \mathbf{x}_i^\top \boldsymbol{\beta}\right)
$$

$$
\boldsymbol{\beta}^{(t+1)} 
= \boldsymbol{\beta}^{(t)} 
+ \alpha \, \mathbf{x}_i 
\left(y_i - \mathbf{x}_i^\top \boldsymbol{\beta}^{(t)}\right)
$$

- Faster, cheaper updates  
- Noisy updates but often converges quicker in practice  


- Very cheap per update  
- Noisy trajectory — never fully settles, oscillates near the minimum  
- Can escape shallow local minima (useful in non-convex problems)

#### 3. Mini-batch gradient descent (MBGD)

A compromise: sample a random mini-batch $\mathcal{B} \subset \{1, \dots, n\}$ of size $m$ at each step:

$$
\nabla_{\boldsymbol{\beta}} L_{\mathcal{B}} 
= \frac{-2}{m}\,\mathbf{X}_{\mathcal{B}}^\top 
\left(\mathbf{y}_{\mathcal{B}} - \mathbf{X}_{\mathcal{B}}\boldsymbol{\beta}\right)
$$

$$
\boldsymbol{\beta}^{(t+1)} 
= \boldsymbol{\beta}^{(t)} 
+ \frac{2\alpha}{m}\,\mathbf{X}_{\mathcal{B}}^\top 
\left(\mathbf{y}_{\mathcal{B}} - \mathbf{X}_{\mathcal{B}}\boldsymbol{\beta}^{(t)}\right)
$$

- Balances stability (BGD) and speed (SGD)  
- Typical batch sizes: $m \in \{32, 64, 128, 256\}$  
- The standard approach in practice  



|                | BGD            | SGD            | Mini-batch      |
|----------------|----------------|----------------|-----------------|
| Gradient estimate | Exact         | Very noisy     | Approximate     |
| Cost per update  | $\mathcal{O}(np)$ | $\mathcal{O}(p)$ | $\mathcal{O}(mp)$ |
| Convergence      | Smooth        | Noisy          | Moderate noise  |
| Memory           | Full dataset  | Single sample  | Batch only      |


### 5. The learning rate $\alpha$

The learning rate is the most critical hyperparameter. It controls the size of each step along the loss surface.

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \, \nabla L
$$

**Too large ($\alpha \uparrow$):** overshoots the minimum — loss diverges or oscillates.  

**Too small ($\alpha \downarrow$):** converges but extremely slowly.  

**Just right:** loss decreases steadily each epoch and levels off near the minimum.

A theoretical upper bound for guaranteed convergence in BGD is:

$$
\alpha < \frac{1}{\lambda_{\max}(\mathbf{X}^\top \mathbf{X}) / n}
$$

where $\lambda_{\max}$ is the largest eigenvalue of $\mathbf{X}^\top \mathbf{X}$. In practice, $\alpha$ is found by search or a learning rate schedule.


### Convergence criterion

Training stops when one of these is satisfied:

$$
\|\nabla_{\boldsymbol{\beta}} L\|_2 < \epsilon \quad \text{(gradient is flat)}
$$

$$
|L^{(t)} - L^{(t-1)}| < \epsilon \quad \text{(loss stopped improving)}
$$

$$
t \geq T_{\max} \quad \text{(maximum iterations reached)}
$$

A common default is $\epsilon = 10^{-6}$.

### Feature scaling

Because gradient descent is sensitive to the scale of each feature, features must be **standardised** before fitting:

$$
\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}
$$

where $\bar{x}_j$ and $s_j$ are the mean and standard deviation of feature $j$ computed on the training set only.

Without scaling, features with large magnitudes dominate the gradient, creating a very elongated loss bowl. The contours of $L$ become elliptical and gradient descent zig-zags slowly toward the minimum instead of descending directly.

### Full algorithm (mini-batch)

$$
\begin{aligned}
&\textbf{Initialise:} \quad \boldsymbol{\beta} \leftarrow \mathbf{0} \; (\text{or small random values}) \\
&\textbf{Standardise:} \quad \mathbf{X} \text{ using training mean and std} \\[6pt]

&\textbf{For } t = 1, 2, \dots, T_{\max}: \\
&\quad \text{Shuffle the training data} \\
&\quad \text{For each mini-batch } \mathcal{B} \text{ of size } m: \\
&\qquad \boldsymbol{\varepsilon}_{\mathcal{B}} \leftarrow \mathbf{y}_{\mathcal{B}} - \mathbf{X}_{\mathcal{B}} \boldsymbol{\beta} \\
&\qquad \nabla L \leftarrow \frac{-2}{m}\mathbf{X}_{\mathcal{B}}^\top \boldsymbol{\varepsilon}_{\mathcal{B}} \\
&\qquad \boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \alpha \nabla L \\[6pt]

&\quad \text{Compute full training loss } L(\boldsymbol{\beta}) \\
&\quad \text{If } \left|L^{(t)} - L^{(t-1)}\right| < \epsilon \text{, break} \\[6pt]

&\textbf{Return } \boldsymbol{\beta}
\end{aligned}
$$

### Why gradient descent converges for linear regression

The MSE loss for linear regression is **strictly convex** in $\boldsymbol{\beta}$ (the Hessian $\frac{2}{n}\mathbf{X}^\top \mathbf{X}$ is positive definite when $\mathbf{X}$ has full column rank). This means:

- There is exactly **one global minimum** — no local minima to get trapped in  
- The loss surface is a **paraboloid** (bowl shape) in parameter space  
- Gradient descent is **guaranteed to converge** to $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ for a sufficiently small $\alpha$  

This is why linear regression is a perfect pedagogical setting for gradient descent — the geometry is clean and the outcome is theoretically guaranteed.

### Summary

| Property               | Expression |
|----------------------|------------|
| Loss                 | $L = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$ |
| Gradient             | $\nabla L = \frac{-2}{n}\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ |
| Update rule          | $\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \alpha \nabla L$ |
| Convergence condition| $\|\nabla L\|_2 < \epsilon$ |
| Loss surface         | Strictly convex — one global minimum |
| Final solution       | $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$ |