---
aliases:
  - training
---
Training of the model is the process of adjusting the parameters of the model based on the input data, on the out put of which is the trained model.

Formally:
	$\mu : (X, Y)^l \to A$
Based on the training set: 
	$X^l=(x_{i}, y_{i})^l_{i=1}$
Calculating the algorithm (adjusting the parameters)
	$a = \mu(X^l)$

$$
\begin{bmatrix}\begin{pmatrix} f_{1}(x_{1})\dots f_{n}x_{1}\\
  \ddots \\
f_{1}(x_{l})\dots f_{n}(x_{l})
\end{pmatrix} \to^y \begin{pmatrix}y_{1}\\ \vdots\\y_{l}\end{pmatrix}\end{bmatrix} \to^\mu a
$$
Here, $a$ - trained algorithm.


Most of the time the training is just an optimization problem.
So, in order to solve it, the **loss function** is needed. The solution to a problem (a parameters for the model) will be a **minimum of a loss function**
$$
\mathcal{L}(\hat{y}, y) = \mathcal{L}(a, x) = \mathcal{L}(g(x, \theta), x)
$$
In this formula $x$ and its $y(x)$ is known constants, so formally we can just write $\mathcal{L}(a, x)$, since the value of loss function depends only on the algorithm with some parameters. The loss function operates on each object from $X^l$.

For example, the most basic loss function for [[Classification]] problem could be:
$$
\mathcal{L}(a, x) = [a(x) \neq y(x) ]
$$
And for [[Regression]], so called absolute error:
$$
\mathcal{L}(a, x) = |a(x) \neq y(x)|
$$
The loss function just indicates how the prediction of model $a$ is different from the actual value $y$.

To measure the quality of predictions on all objects $X^l$ the function called **risk**:
$$R(a) = E[\mathcal{L}(a, x)]$$
Due to the fact that it is impossible to find the true expected value, because the distribution of $\mathcal{L}(a, x)$ is unknown, the **Empirical risk** is calculated:
$$
R_{emp}(a, X^l) = \frac{1}{l} \sum_{i=1}^l\mathcal{L}(a, x_{i})
$$
**Empirical risk minimisation (ERM)**:
Previously, in the formal definition of training the $\mu$ was the training process. Finding  minimum of Empirical risk is the training process (finding parameters) in the method of empirical risk minimisation:
$$
\mu(X^l) = {\arg\!\min_{a \in A}}\text{ }R_{emp}(a, X^l)
$$

Example: Least squares
	The problem is to find a model for **linear regression**:
		$g(x, \theta) = \sum _{j=1}^n \theta_{j} f_{j}(x)$
	The **loss function** here is:
		$\mathcal{L}(a, x) = (a(x) - y(x))^2$
	**Emperical risk**:
		$R_{emp}(a, X^l) = \sum_{i=1}^l\mathcal{L}(a, x_{i}) = \sum_{i=1}^l(g(x_{i}, \theta) - y(x))^2$
	The training result:
		$\mu(X^l)={\arg\!\min_{a \in A}}\text{ }\sum_{i=1}^l(g(x_{i}, \theta) - y(x))^2)$

This method is most widely used in [[Supervised learning|supervised learning]]. But some other models use other methods for training.