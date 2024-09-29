Logistic regression is a model used to solve [[Classification]] problems, is a type of [[Linear classifiers]].

### Two classes

$Y = \{ -1 , +1 \}$

The model $a$:
$$
\begin{array}{}
a(x) = sign\langle w, x\rangle \\
x,w \in \mathbb{R}
\end{array}{}
$$
Margin: 
$$
M= \langle w, x\rangle y
$$
Logarithmic loss-function (***log-loss***)
$$
\mathcal{L}(M) = \log(1+e^{-M})
$$
From [[Probabilistic models]]:
$$
\begin{array}{l}
-\log P(y_{i}|x_{i}, w) = \mathcal{L}(M)) \\
e^{-\log P(y_{i}|x_{i}, w)} = e^{\log(1+e^{-M})} \\
\frac{1}{P(y_{i}|x_{i}, w)} = 1+e^{-M} \\
\implies \\
P(y_{i}|x_{i}, w) = \frac{1}{1+e^{-M}}
\end{array}{}
$$
$\frac{1}{1+e^{-M} } =  \sigma(M)$ - sigmoid function
	$\sigma(M) + \sigma(-M) = 1$

The sigmoid function is a correct function to describe $P(y_{i}|x_{i}, w)$ since $\sigma(M): [-\infty; \infty] \to [0; 1]$

Likelihood maximisation:
$$
\begin{array}{l}
L(x) = \sum_{i=1}^l \log P(y_{i}|x_{i}, w) \to \max_{w} \\
\text{Substituting with } \sigma(M) \\
= \sum_{i=1}^l \log(\sigma(M)) \to \max_{w} \\
= \sum_{i=1}^l \log(\frac{1}{1+e^{-M} }) \to \max_{w} \\
= -\sum_{i=1}^l \log(1+e^{-M}) \to \max_{w}
\end{array}{}
$$
Final minimisation function:
$$
R_{emp} (x, w)= \sum_{i=1}^l \log(1+e^{-\langle w, x_{i}\rangle y_{i}}) \to \min_{w}
$$
