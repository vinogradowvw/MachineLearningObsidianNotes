This is one of the first mathematical "Machine Learning" model proposed in 1943 by Warren McCulloch and Walter Pitts. It is a type of [[Linear classifiers]] which make predictions based on linear combination of weights with futures.

![[perceptron.png]]

Formally:
$$
a(w, x) = \sigma(\langle w, x_{i}\rangle) = \sigma(\sum_{j=1}^nw_{j}f_{j}(x) -w_{0})
$$
	$\sigma$ is the activation function, f.e. - $sign$
	$w_{j}$ - weights
	$w_{0}$ - **bias unit**

**ERM**:
$$
R_{emp} = \sum_{i=1}^l\mathcal{L}_{i}(w) \to \min_{w}
$$
It is is important that ERM has exactly the sum, not the average and $\mathcal{L}$ is continuous in the formula.

To find a minimum of this type of ERM is possible with [[Gradient descend methods]].