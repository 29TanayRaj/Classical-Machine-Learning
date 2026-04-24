# Linear Regression — Parameter Estimation via Gradient Descent

## 1. Motivation

The closed-form OLS solution:

$$
\hat{\beta} = (X^\top X)^{-1} X^\top y
$$

- Requires inverting a $(p+1) \times (p+1)$ matrix  
- This becomes computational costly: $O(p^3)$  
- Expensive when $p$ is large.

Idea is to use gradient descent to iteratively minimize the loss.


## 2. The Loss Function

Mean Squared Error (MSE):

$$
L(\beta) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

Matrix form:

$$
L(\beta) = \frac{1}{n} \|y - X\beta\|^2
$$

> Note : Some formulations use $\frac{1}{2n}$ to make the derivative calculation cleaner. This fractor does not affect the location of minima, has effect only on the learning rate for the process. 


## 3. The Gradient

Expanded loss:

$$
L(\beta) = \frac{1}{n} \left( y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X \beta \right)
$$

Gradient:

$$
\nabla L(\beta) = \frac{-2}{n} X^\top (y - X\beta)
$$


### Intuition

- Residual:  
  $$
  \varepsilon = y - X\beta
  $$

- Gradient = projection of residuals onto feature space  
- At optimum:  
  $$
  X^\top \varepsilon = 0
  $$  
  (normal equations)

## 4. Gradient Descent

Update rule:

$$
\beta^{(t+1)} = \beta^{(t)} - \alpha \nabla L(\beta^{(t)})
$$

Substitute gradient:

$$
\beta^{(t+1)} = \beta^{(t)} + \frac{2\alpha}{n} X^\top (y - X\beta^{(t)})
$$

Where:
- $\alpha > 0$, is the learning rate. 


## 4.1 Batch Gradient Descent (BGD)

- The idea id to use all $n$ samples to compute the gradient of the loss function, then use this to update the parametrs using the loss function. 

Advantages: 
- Stable  

Disadvantage: 
- Expensive  
- Slower in comparision to other methods. 
- Not good for non-convex loss functions (not the case with squared error loss though).


## 4.2 Stochastic Gradient Descent (SGD)

Per sample $(x_i, y_i)$:

$$
\nabla L_i = -2 x_i (y_i - x_i^\top \beta)
$$

Update:

$$
\beta^{(t+1)} = \beta^{(t)} + 2\alpha x_i (y_i - x_i^\top \beta)
$$

- Fast  
- Noisy  

---

## 4.3 Mini-batch Gradient Descent (MBGD)

For batch $B$ of size $m$:

$$
\nabla L_B = \frac{-2}{m} X_B^\top (y_B - X_B \beta)
$$

Update:

$$
\beta^{(t+1)} = \beta^{(t)} + \frac{2\alpha}{m} X_B^\top (y_B - X_B \beta)
$$

---

### Comparison

| Method | Gradient | Cost | Noise |
|--------|--------|------|------|
| BGD | Exact | High | Low |
| SGD | Noisy | Low | High |
| MBGD | Approx | Medium | Medium |

Typical batch sizes: $32, 64, 128, 256$

---

## 5. Learning Rate ($\alpha$)

$$
\beta^{(t+1)} = \beta^{(t)} - \alpha \nabla L
$$

- Too large → divergence  
- Too small → slow  

Condition:

$$
\alpha < \frac{1}{\lambda_{\max}(X^\top X)/n}
$$

- A better approach will be to use a adaptive learning rate which is a function of iterations, can change (decrease as the iterations increase). 

- Note for self: What if the adaptive learing rate is a function of error?? how will that affect the training process? 


## 6. Convergence Criteria

Stop when:

 $$ \|\nabla L\|_2 \le \epsilon $$
 $$ |L(t) - L(t-1)| \le \epsilon $$
- or when Maximum iterations reached  


## 7. Feature Scaling

Standardization:

$$
\tilde{x}_j = \frac{x_j - \mathrm{mean}(x_j)}{\mathrm{std}(x_j)}
$$

Why:

- Prevent domination by large features, if one of the features is large in scale it can have a greater influence during the gradient update. 
- Faster convergence on large scale data mostly beacuse of memory consumption during computation. 


## 8. Full Algorithm (Mini-batch)

1. Initialize $\beta = 0$  
2. Standardize $X$  
3. For $t = 1,2,\dots,T$:
   - Shuffle data  
   - For each batch $B$:
     - Residual:
       $$
       r = y_B - X_B \beta
       $$
     - Gradient:
       $$
       \nabla L = \frac{-2}{m} X_B^\top r
       $$
     - Update:
       $$
       \beta \leftarrow \beta - \alpha \nabla L
       $$
4. Check convergence  
5. Return $\beta$  


## 9. Why Gradient Descent Converges

Loss:

$$
L(\beta) = \frac{1}{2n}\|y - X\beta\|^2
$$

- Hessian:
  $$
  \frac{1}{n} X^\top X
  $$
- Positive definite (if full rank)

One global minimum, guaranteed convergence

---

## 10. Summary

Loss:

$$
L = \frac{1}{n}\|y - X\beta\|^2
$$

Gradient:

$$
\nabla L = \frac{-2}{n} X^\top (y - X\beta)
$$

Update:

$$
\beta \leftarrow \beta - \alpha \nabla L
$$

Solution:

$$
\hat{\beta} = (X^\top X)^{-1} X^\top y
$$