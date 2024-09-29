---
aliases:
  - gradient descend
  - GD
  - SDG
  - SAG
  - Gradient
---
# Gradient descend (GD)
The gradient descend method is used to calculate the minimum of [[Training|empirical risk]] with continuous [[Training|loss function]].

Let $w$ will be the vector of weights (parameters) of the model.

The $w^{(0)}$ - **bias** is set.
Then:

$$w^{(t+1)} = w^{(t)} - h \nabla R_{emp}(w^{(t)})$$

Here:
	$h$ - **learning rate**
	$\nabla R_{emp}(w^{(t)})=\left(  \frac{\partial R_{emp}(w^{(t)})}{\partial w_{j}^{(t)}} \right)_{j=0}^n = \sum_{i=1}^l \nabla \mathcal{L}(w^{(t)})$ - Gradient of empirical risk
It follows:
$$w^{(t+1)} = w^{(t)} - h  \sum_{i=1}^l \nabla \mathcal{L}(w^{(t)})$$
Here, the gradient is calculated on all objects on one time and only then the vector of weights $w^{(t)}$ is updated to $w^{(t+1)}$

# Stochastic Gradient Descent (SGD)
The main idea behind the **stochastic gradient descend** is not to calculate the gradient with all $l$ objects at once, but by calculation of gradient on $i$th object update weights immediately. With this method the calculation of weights and ERM will be faster and less expensive.
The object from which new weights will be calculated is chosen randomly thus this algorithm called **stochastic gradient descend**.

**Stochastic Gradient Descend algorithm**:
1. Weights $w$ are somehow initialised.
2. Define the empirical risk function as **mean** of loss function on all object:
		$R_{emp}(w) = \frac{1}{l} \sum_{i=1}^l\mathcal{L}(w)$
3. While $R_{emp}$ or/and $w$ have not converged, repeat the following steps:
	Randomly choose $x_{i} \in X^l$;
	Calculate loss: 
		$\varepsilon_{i} = \mathcal{L}_{i}(w)$;
	Make a gradient step:
		 $w := w-h\nabla \mathcal{L}_{i}(w)$
	Calculate new value of $R_{emp}$ using exponential smoothing:
		$R_{emp} := \lambda \varepsilon_{i} + ( 1-\lambda)R_{emp}(w)$;

# Stochastic Average Gradient (SAG)

This gradient descend algorithm is very similar to SDG but has 2 new steps.

**Stochastic Average Gradient algorithm**:
1. Weights $w$ are somehow initialised.
2. <span style="color:red">Vector of gradients is initialised (values of loss function gradients with default weights for each</span> $x_{i}$):
		$G^l: G_{i} = \nabla \mathcal{L}_{i}(w)$
3. Define the empirical risk function as **mean** of loss function on all object:
		$R_{emp}(w) = \frac{1}{l} \sum_{i=1}^l\mathcal{L}(w)$
4. While $R_{emp}$ or/and $w$ have not converged, repeat the following steps:
	Randomly choose $x_{i} \in X^l$;
	Calculate loss: 
		$\varepsilon_{i} = \mathcal{L}_{i}(w)$;
	Calculate $G_{i}$
	Make a gradient step with <span style="color:red">average of all gradients:</span>
		 $w := w-h \frac{1}{l}\sum_{i=1}^l G_{i}$
	Calculate new value of $R_{emp}$ using exponential smoothing:
		$R_{emp} := \lambda \varepsilon_{i} + ( 1-\lambda)R_{emp}(w)$;

Another approach is to calculate the average of gradients with **exponential smoothing**:
	$momentum := \gamma \cdot momentum + (1-\gamma)\nabla \mathcal{L}_{i}(w)$
	$w := w-h\cdot momentum$

# Weights initialisation

There is some simple ways of weights initialisation such as:
	$\forall i: w_{j} = 0$
	$\forall i: w_{j} = random\left( -\frac{1}{2n}; +\frac{1}{2n} \right)$
But in case of initialisation with those methods can cause some problem with GD, f.e. $R_{emp}$ is converged to local but not global minimum.

Other methods can be:
	$w_{j} := \frac{\langle y, f_{j}\rangle}{\langle f_{j}, f_{j}\rangle}$
		optimal if the loss function is quadratic (method comes from MMSE) and $f_{j}$ are not correlated.
	Get weights by learning from subset of objects.
	Multistart: GD with different start $w$ and choose the best.

# Newton-Raphson method

The main idea behind the Newton-Raphson method is to make the GD faster by calculating the Hessian and normalise the learning rate by it's diagonal value:
	 $$w := w-h \left( \frac{\partial^2\mathcal{L}_{i}(w)}{\partial w_{j} \partial w_{j}} + \mu \right)^{-1} \nabla \mathcal{L}_{i}(w)$$
	Here  $\mu$ is constant in case if $\left( \frac{\partial^2\mathcal{L}_{i}(w)}{\partial w_{j} \partial w_{j}} + \mu \right)$ will be 0.

This method make the GD go faster to minimum when the function if far from it and slowing down the descend as it goes to the minimum.