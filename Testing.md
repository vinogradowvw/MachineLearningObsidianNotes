---
aliases:
  - overfitting
  - testing
  - split
  - train-test-split
  - cross-validation
  - loave-one-out
---
Testing is the process of evaluating the performance of a trained model on new data.

 After training, we have a model $a$ with parameters $\theta$:
	 $a = \mu(X^l)$, $X^l$ - training set
Let $X^t = \{ x_{i} \}^t_{i}$ be the test set with known $y(x_{i}^t): X\to Y$ for every $x_{i}^t$
Then we can calculate the loss function and empirical risk on test set.
The overall performance could be measured by the empirical risk:
$$
R_{emp}(a, X^t) = \frac{1}{t} \sum_{i=1}^t\mathcal{L}(a, x_{i})
$$

There is also other **metrics** for evaluating the performance of a trained model that is **not** connected with the loss function, but still useful, for example: F1 and ROC AUC score for classification.

# Overfitting and underfitting

The overfitting is the state when the model predicts too close to the data that it was trained on. The goal of the machine learning models is not just to predict the closest value of $Y$ to the true one. The goal is to find a pattern or to generalise the relationship between $X$ and $Y$. Thus overfitting is not good.

Generally, if $R_{emp}(a, X^t) \gg R_{emp}(a, X^l)$, the error of prediction is far less on training set than on test set, than the model is overfitted.

Most of the time, the overfitting is caused by high dimensionality of the model.

Underfitting is the opposite from the overfitting. It means that the model is not predicting well due to the lack of it's complexity.
![[underfitting-overfitting.png]]
It is not possible to get rid of overfitting completely, but it is possible to minimise it.

There is multiple techniques how to make a testing, which allows to notice that model is overfitted.

1.  **Hold-out**:
		Choosing multiple objects for test set independently from train set:
			$HO(\mu,X^l, X^t) = R_{emp}(\mu(X^l), X^t) \to min$
		- This method depends on randomness of splitting the data on train and test.
2. **Leave-one-out**:
		Let $L$ be number of objects we have in the set. $L$ times the model is trained on all but 1 singe object. and then validated on this one left-out object. Than the mean of this measures is calculated.
			$LOO(\mu, X^L) = \frac{1}{L}\sum_{i=1}^L\mathcal{L}(\mu(X^L/x_{i}),x_{i} \to min$
		This method is stable, and not dependant on randomness, but is computationally expensive, due to the fact that the model will be trained $L$ times.
3. **Cross-validation**
		The idea is to make Leave-one-out less computationally expensive:
		Split the whole data set $X$ into $n$ samples (let $N$ be a set of samples). The model is then trained $N$ times, each time excluding one sample and using the remaining data for training. The performance is evaluated on the excluded sample by calculating a loss measure. Finally, the average of these loss values is computed to assess the overall performance.
			$CV(\mu, X) = \frac{1}{n}\sum_{i=0}^{|N|}\mathcal{L}(\mu(X/N_{i}), N_{i}) \to min$

By minimising any of this metrics the overfitting effect is minimised.