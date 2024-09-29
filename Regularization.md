**Regulatization** is a method to reduce complexity of model, reduce an effect of multicollinearity by lowering weights of [[Model]].

To implement this the [[Training|loss function]] is modified.
$$
\tilde{{\mathcal{L}_{i}}}(w) = {\mathcal{L}_{i}}(w) + \frac{\tau}{2}||w||^2 = {\mathcal{L}_{i}}(w) + \frac{\tau}{2} \sum_{j=1}^n w_{j}^2 \to min
$$
[[[Gradient descend methods|Gradient]]:
$$
\nabla \tilde{{\mathcal{L}_{i}}}(w) = \nabla {\mathcal{L}_{i}}(w) + {\tau} w
$$
The [[Gradient descend methods|gradient descend]] step is now:
$$
w := (w-\tau h)-h\nabla \mathcal{L}_{i}(w)
$$
### L2 and L1

The above example was so called $L_{2}$ regularization (**Ridge**):
$$
\tilde{{\mathcal{L}_{i}}}(w) = {\mathcal{L}_{i}}(w) + \frac{\tau}{2}||w||^2 = {\mathcal{L}_{i}}(w) + \frac{\tau}{2} \sum_{j=1}^n w_{j}^2 \to min
$$
This approach penalizes the sum of **squared** weights, encouraging smaller weight values without driving them to zero.

$L_{1}$ (**Lasso**) regularization penalizes the sum of **absolute** weights, thus can drive some weights to **exact zero**, effectively performing **feature selection** by removing less important features from the model.
$$
\tilde{{\mathcal{L}_{i}}}(w) = {\mathcal{L}_{i}}(w) + \tau||w|| = {\mathcal{L}_{i}}(w) + \tau \sum_{j=1}^n |w_{j}| \to min
$$


