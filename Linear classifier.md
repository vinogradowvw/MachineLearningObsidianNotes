Linear classifier solving classification problem with following model:
(For simplicity classification into 2 groups will be considered)
$$g(x, \theta) = sign \sum _{j=1}^n \theta_{j} f_{j}(x); \text{(sign of summation of all features with coefficients)}, Y = \{ -1, +1 \}$$
or
$$
a(x, w) = sign\langle w, x\rangle = sign \sum_{j=1}^nw_{j}f_{j}(x) \text{   - sign of linear combination of weights and features}
$$
#### **Loss function:**

If the classes are {-1 , +1} than [[Training|loss function]] in typical case could look like:
	$\mathcal{L}(a, x)= [a(x)y(x) < 0]\text{  (same sign -> + in result)} = [\langle w, x\rangle y < 0]$
![[margin.svg]]
$\langle w, x\rangle y$ - so called **margin** - distance between the object and the dividing hyperplane. By this measure is also possible to tell how far is the object from the divider and how "confident" the prediction was.

It follows that the basic loss function can be substituted or approximated by the function of margin, which will be continuous, and not the step function as it was.
$$
\mathcal{L}(a, x)= [a(x)y(x) < 0] = [\langle w, x\rangle y < 0] \ll \mathcal{L}(\langle w, x\rangle y)
$$
Margin could be calculated not only with linear classifiers. This is very fundamental term in classification. Margin can be generalised as:
	$M_{i}(w) = g(w, x_{i})y_{i}$, where $g(w, x_{i})$ is any function model function but without "sign"
#### **ERM:**
With new monotonic loss function [[Training|empirical risk]] is:
$$
R_{emp} (w) = \sum_{i=1}^l[\langle w, x_{i}\rangle y_{i} < 0] \ll \sum_{i=1}^l\mathcal{L}(\langle w, x_{i}\rangle y_{i}) \to min
$$
It is also possible to approximately minimise the **empirical risk** with continuous function.


Some models with **continuous loss function**:
	- [[Support Vector Machine]]
	- [[Logistic Regression]]
	- Neural Networks with sigmoid activation function
	- [[AdaBoost]] (exponential loss function)