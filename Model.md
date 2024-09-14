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
