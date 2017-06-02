# ML_Stanford_University
Andrew Ng's machine learning course at Stanford, available through Coursera

## Week 4: Neural Networks
### Neurons are computational units that take in inputs via dendrites and output electricity through axons
1. The input layer is comprised of or features in the design matix (x1, x2, ..., xn) similar to the dendrites
2. The ouptut layer gives us the result of the hypothesis function (activation function). Neural networks use the simgmoid function seen earlier in logistic regression.
3. The hidden layers are between the input and output layers of the model and contain the activation units. Each of the activation unit contains an expression made up of our input features and parameters (weights).
4. Each parameter (j) is a matrix of weights controlling function mapping from layer j to layer j + 1
5. Dimensions of weight matrices are determined like so:
	* If network has s(j) units in layer j and s(j+1) units in layer j + 1, then Theta(j) will be of dimension s(j+1) x (s(j) + 1)

## Week 5: Backpropagation
1. Neural networks have many output nodes, to account for this the cost function associated with neural networks contain nested summations that loop through the number of output nodes.
2. The backpropagation algorithm is as follows:
	set D = 0 for all l, i, j
	For i = 1 to m in training set {x(i), y(i)}
		Set a(l) (a at layer l) = x(i)
		Preform forward propagation to compute a(l) for l=2,3,...,L
		Using yi, compute S(l) = a(l) - y(i)
		Compute S(l - 1), S(l - 2), ..., S(2)
		Delta(l)(ij) = Delta(l)(ij) + a(l)(j) * S(l + 1)
	Delta(l)(ij) = (1 / m) * Delta(l)(ij) + Lambda * Theta(l)(ij) if j (not equal) to 0
	Delta(l)(ij) = (1 / m) * Delta(l)(ij) if j= 0

Partial Derivative of Theta(l)(ij) * J(Theta) = Delta(l)(ij)

3. S(j)(l) is the error for the a(j)(l), these delta values are equivalent to the derivative of the cost function. S(l) is computed by subtracting the value(s) of the units in the last layer of the network from the actual results (y).
4. Each subsequent value of S(l-1) is calculated by multiplying S(l) and the theta values associated with the layer l - 1
5. In order to check if your implementation of backpropagation is correct you can approximately compute the gradient using (J(Theta + e) - J(Theta - e))/2e with an e value of 10^(-4) (for example). If our values for Delta are approximately equal to our gradient approximation values the backpropogation has most likely been implemented correctly.
6. The values of Theta cannot be initialized to 0 because then all of the activation functions would evaluate to the same value and the activation units in subsequuent layers would evaluate to the same values as well. This symetry also would hold for the values of S across all layers, resulting in a network producing the same values for all layers for each iteration of propagation.
7. In order to break this symetry across layers, random initialization is used to compute the starting values of Theta between -e and e (where e is some small value epsilon, independent of the epsilon used in gradient approximation) i.e.
```octave
Theta1 = rand(10, 11) * (2 * initEpsilon) - initEpsilon
```
In which rand is a function in octave that will initialize a matrix of random real numbers between 0 and 1
