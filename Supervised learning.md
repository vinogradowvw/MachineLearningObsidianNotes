---
aliases:
  - supervised learning
  - Supervised learning
  - feature
---
The supervised learning is a category in machine learning that uses labeled data to train models to predict outcomes on not labeled objects.

# The problem
$X$ - the set of objects
$Y$ - the set of labels

The goal of supervised learning is to find the model (function) that will generalise the following unknown relationship ("***target function***"):
	$y: X\to Y$


The problem of the supervised learning typically looks like this:
Given:
	$\{ x_{1},x_{2},x_{3}\dots x_{n}\} \subset X \}$ - ***training sample*** (***training set***)
	$y_{i}=y(x_{i})$ - known labels
	in other words: $\{(x_{i}, y_{i})\}$ - the set of pairs of $x$ and $y$
 Find:
	 $a: X \to Y$ - an algorithm (function) that will map the object to it's label.


The difference from just approximation is that the object $X$ is usually not just a number on $\mathbb{R}$, it is complicated structure of information.

# The definition of the object
The object is defined by it's features (properties of object):
	$f_{j}: X \to D_{j}$

There are several types of features:
	$D_{j} = \{ 0,1 \}$ - ***binary feature*** (ex. Watcher: "Subscribed ti channel" - yes/no)
	$D_{j}: |D_{j}| < \infty$ - ***nominal feature*** (ex. Color: {Red, Blue, Green, Yellow} )
	$D_{j}: |D_{j}| < \infty; D_{j}$ - *ordered set*  - ***ordinal feature*** (ex. Likert scale)
	$D_{j} = \mathbb{R}$ - ***numerical feature*** (ex. Income)

Each object has a vector:
	$(f_{1}(x), f_{2}(x), f_{3}(x)\dots f_{n}(x))$ - the vector of features.

It follows that all objects in the set create the **future data matrix**.
	$$
	F = ||f_{j}(x_{i})||_{l\times n} = \begin{pmatrix} f_{1}(x_{1})\dots f_{n}x_{1}\\
  \ddots \\
f_{1}(x_{l})\dots f_{n}(x_{l})
\end{pmatrix}
	$$
This matrix represents the **data set** that we usually use. Each row is the object $X$, and each column is the feature $D_{j}$  $\implies$ $f_{j}(x_{i})$ is a feature $j$ of object $i$