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
	![J(\Theta) = \frac{-1}{m} \sum_{i=1}^m \sum{k=1}^K](http://latex.codecogs.com/gif.latex?Concentration%3D%5Cfrac%7BTotalTemplate%7D%7BTotalVolume%7D)
