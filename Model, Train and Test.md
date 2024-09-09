# Model

Model (machine learning or predictive model or algorithm) is a parametric function family:

	$A = \{ g(x, \theta)| \theta \in \Theta \}$
		Here, $\theta$ is a parameter (vector (combination) of parameters) of model and $\Theta$ is a parameter space (all possible combinations of parameters).
	$g: X \times \Theta \to Y$ - is a concrete function (model) that predicts $Y$

Example:
	Linear model with vector of parameter $\theta = \{ \theta_{1}, \theta_{2}, \theta_{3}, \dots \theta_{n} \}, \Theta = \mathbb{R}^n$
[[Regression]]
$$g(x, \theta) = \sum _{j=1}^n \theta_{j} f_{j}(x); \text{(summation of all features with coefficients)}, Y = \mathbb{R}$$
[[Classification]]
$$g(x, \theta) = sign \sum _{j=1}^n \theta_{j} f_{j}(x); \text{(sign of summation of all features with coefficients)}, Y = \{ -1, +1 \}$$

# Training
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

# Testing
The testing is a process where by putting new data into model, on the outcome having the prediction.
// todo