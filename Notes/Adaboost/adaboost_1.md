## AdaBoost

#### Introduction

AdaBoost is an ensemble learning technique that combines multiple weak learners to produce a strong learner. The idea is to iteratively train weak learners. After each iteration, the training examples are reweighted so that observations misclassified by the previous learner receive more attention during the next iteration.

The main question is how we keep track of these errors. AdaBoost does this by assigning weights to the training examples (observations). If a training example is misclassified, its weight is increased so that the next weak learner pays more attention to it during training.

AdaBoost is short for Adaptive Boosting because the algorithm adapts by recalculating the weights assigned to the training observations after every iteration. One could also argue that Additive Boosting would be an appropriate name because each iteration adds a new learner to the ensemble, with its contribution determined by its classification error.

The contribution of each learner in the ensemble is determined by its weighted classification error. A learner with a lower error receives a larger weight, while a learner with a higher error contributes less to the final prediction.

Here is a algorithm for adaboost, taken from "Elements of Statistical Learning".

### Algorithm

### Algorithem 

$
\begin{aligned}
\quad \textbf{Input: } & \{(x_i, y_i)\}_{i=1}^{N}, \quad M \text{ boosting rounds} \\
\\
1.\;& \textbf{Initialize observation weights:} \\
& w_i \leftarrow \frac{1}{N}, \qquad i = 1,2,\ldots,N. \\
\\
2.\;& \textbf{For } m = 1 \text{ to } M: \\
& \quad \text{(a) Fit a classifier } G_m(x) \text{ using weights } w_i. \\
\\
& \quad \text{(b) Compute the weighted training error:} \\
& \qquad
\mathrm{err}_m =
\frac{\sum_{i=1}^{N} w_i\,\mathbf{1}\!\left(y_i \neq G_m(x_i)\right)}
{\sum_{i=1}^{N} w_i}. \\
\\
& \quad \text{(c) Compute the classifier weight:} \\
& \qquad
\alpha_m =
\log\left(\frac{1-\mathrm{err}_m}{\mathrm{err}_m}\right). \\
\\
& \quad \text{(d) Update the observation weights:} \\
& \qquad
w_i \leftarrow
w_i \exp\!\left(
\alpha_m\,
\mathbf{1}\!\left(y_i \neq G_m(x_i)\right)
\right),
\qquad i = 1,\ldots,N. \\
\\
3.\;& \textbf{Output:} \\
& \qquad
G(x) =
\operatorname{sign}
\left(
\sum_{m=1}^{M}
\alpha_m G_m(x)
\right).
\end{aligned}
$

> Note 1: One thing to keep in mind that we are dealing with two kinds of weights here, both of which are calculated: $w_i$'s for training weights, and $\alpha_m$'s for the model weights.

> Note 2: In practice the normalization in the step 2.b is done in the step 2.d, this is done to inform the weak learner about the misclassified examples.