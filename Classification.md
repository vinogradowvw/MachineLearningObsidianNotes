The classification is the type od [[Supervised learning|supervised learning]] problem where the goal is to predict which class object is assigned to.

Given:
	$\{ x_{1},x_{2},x_{3}\dots x_{n}\} \subset X \}$ - ***training sample*** (***training set***)
	$y_{i}=y(x_{i})$ - known labels
Find:
	 $a: X \to Y$ - an algorithm (function) that will map the object to it's label.
	 Where Y:
		 $Y=\{-1,+1\}$ - 2 classes classification (or in case $Y \in \{0, 1 \}$ - Binary classification)
		 $Y = \{ C_{1},C_{2}\dots C_{M} \}$ - classification into $M$ classes, where classes do not overlap.
		 $Y = \{ 1, 0 \}^M$ - classification into $M$ classes, which can overlap.
