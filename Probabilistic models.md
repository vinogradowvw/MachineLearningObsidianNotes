The training data is generated from an underlying random process, defined over a probability space:
$$
X \times Y - \text{probability space with density:} f(x, y|w) = P(y|x, w)f(x)
$$
$$
\begin{array}{l}
X^l \sim IID \\
(x_{i}, y_{i})_{i=1}^l \sim f(x, y | w)
\end{array}
$$
The objective is to construct a parametric model that approximates the density function over $X \times Y$, which governs the data generation process. The goal is to find the parameters $w$ that maximise the likelihood of the observed data, ensuring that the model best captures the underlying distribution.

Maximum likelihood estimator is used to find an answer:
Since $X^l \sim IID$, the joint probability density can be expressed as:$f(x,y | w) = \prod_{i=1}^l p(x_{i}, y_{i}|w)$
$$
\prod_{i=1}^l p(x_{i}, y_{i}|w) = \prod_{i=1}^l P(y_{i} | x_{i}, w) \cancel{f(x_{i})} \to max
$$
The critical term here is $P(y_{i} | x_{i}, w)$ representing the probability of the dependant variable $y_{i}$, given the object (feature vector) $x_{i}$ and parameters $w$. Informally, this term essentially captures the **model's prediction** and $P(y | x, w)$ is the actual **model**. On the other hand, $f(x_{i})$ which is independent of $w$. Hence, it is not directly relevant in the context of maximising the likelihood for a supervised learning problem.

In practice, the log-likelihood is used, since it is monotonic.
$$
L(w) = \sum_{i=1}^l \log P(y_{i}|x_{i}, w) \to \max_{w}
$$
### Maximum likelihood and [[Training|loss function]]
Maximum likelihood:
$$
L(w) = \sum_{i=1}^l \log P(y_{i}|x_{i}, w) \to \max_{w}
$$
Empirical risk and loss function:
$$
R_{emp}(w) = \sum_{i=1}^l\mathcal{L}(y_{i}, g(x_{i}, w)) \to \min_{w}
$$
These two principles are equivalent if:
$$
-\log P(y_{i}|x_{i}, w) = \mathcal{L}(y_{i}, g(x_{i}, w))
$$
It follows that the $\log P(y_{i}|x_{i}, w)$ can be used as loss function, under the assumption that the data is generated from an underlying probability distribution.

### [[Regularization]]
Given:
	$P(y | x, w)$ - probabilistic model
Assuming that parameters are not constant but random as well:
	$f(w; \gamma)$ - distribution of $w$, 
		$\gamma$ - so called ***hyperparameters*** - not a random value

In this case, not only the data $X^l$ is randomly generated, but parameters $w$ are also assumed to be random. Thus, the joint distribution of the observed data and the parameters is defined as:
$$
f(X^l, w) = f(X^l|w)f(w; \gamma)
$$
It follows that it is possible to do regularization just by adding $f(w; \gamma)$ into likelihood formula:
$$
L(w) = \sum_{i=1}^l \log P(y_{i}|x_{i}, w) + \log f(w; \gamma)\to \max_{w}
$$
Example:
Assuming that $w \sim N(0, C)$:
$$
\begin{array}{l}
f(w; C) = \frac{1}{\sqrt{ C 2\pi }^n}\exp\left( -\frac{||w||^2}{2C} \right) \\
\implies \\
-\ln f(w; C) = \frac{1}{2C} ||w||^w + \cancel{const}
\end{array}{}
$$

Here $L_{2}$ regularizaion is used and $\frac{1}{C}$ is $\tau$, which generally means that the $\tau$ coefficient reducing the variance of $w$ distribution.