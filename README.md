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
2. Lower case delta sub j in layer l is the error for the activation function j in layer l, these delta values are equivalent to the derivative of the cost function. Lower case delta l is computed by subtracting the value(s) of the units in the last layer of the network from the actual results (y).
3. Each subsequent value of lower case delta (lower case delta in layer l - 1) is calculated by multiplying lower case delta in layer l and the theta values associated with the layer l - 1
4. In order to check if your implementation of backpropagation is correct you can approximately compute the gradient using (J(T + e) - J(T - e))/2e with an e value of 10^(-4) (for example). If our values for Delta are approximately equal to our gradient approximation values the backpropogation has most likely been implemented correctly.
