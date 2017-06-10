# ML_Stanford_University
Andrew Ng's machine learning course at Stanford, available through Coursera

## Week 1: What is Machine Learning?
1. "The field of study that gives computers the ability to learn without being explicitly programmed." -Arthur Samuel
2. "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." -Tom Mitchell
3. Supervised learning
	1. a data set of inputs to corresponding outputs is provided in order to "learn"
	2. Can be regression problems, in order to predict continuous output values i.e. predicting housing prices given square footage, location, etc.
	3. They can also be classification problems in order to predict discrete output values. i.e. Determining whether a tumor is benign or malignent based features taken from diagnosed patient data

4. Unsupervised learning
	1. Uses clustering algorithms to separate data into specific categories without telling the program what the categories are in advance. The algorithm should "learn" how to organize the data i.e. digital signal processing
	2. This clustering process is shown using the Cocktail party algorithm. Andrew used this algorithm to separate a person's voice from background music.
5. Model Representation (supervised learning)
	1. each input feature x(i) is mapped to a corresponding output y(i)
	2. each {x(i), y(i)} is a training example from a training set {{x(1), y(1)}, {x(2), y(2)}, ..., {x(i), y(i)}}, where m represents the number of training examples in the set
	3. Using the training set, the goal is to produce a hypothesis function h(x) that accurately produces new ouputs from new input values not contained within the training set.
	4. If the model is predicting continuous values, the learning algorithm is used to solve a regression problem
	5. If the model is predicting discrete values, the learning algorithm is used to solve a classification problem
6. Cost Function
	1. The cost function is used to train our weights, theta, used in the hypothesis function
	2. in the case of linear regression the cost function should choose values of theta that minimizes the distance between a line drawn through the plotted data and all of the data points on the graph
	3. The goal in this senario is to minimize the squared error value (average error) in (h(x(i)) - y(i))^2
	4. the squared error is measured over all of the input values using the cost function: $$J(\theta) = \sum_{i=1}^m frac{1}{2m}(h(x) - y)^2)$$

## Week 4: Neural Networks
### Neurons are computational units that take in inputs via dendrites and output electricity through axons
1. The input layer is comprised of the features (columns) in the design matrix (X(:,1), X(:,2), ..., X(:,n)); analogous to the dendrites of a neuron.
2. The ouptut layer gives us the result of the hypothesis function (also known as the activation function); analogous to the axons of a neuron. Neural networks use the simgmoid function to calculate this value as seen earlier in logistic regression.
3. The hidden layers in the network exist between the input and output layers of the network and contain the activation units. Each activation unit contains an expression made up of the input features of a training example (a row from the design matrix, X) and weights (a row from the matrix Thetaj).
4. Each parameter j (Theta1, Theta2, ..., Thetaj) is a matrix of weights controlling activation function mapping from layer j to layer j + 1
5. Dimensions of weight matrices Thetaj are determined like so:
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

8. Picking a neural network architechture:
	* Default being 1 hidden layer, and if you have more than one hidden layer it is recommended that you have the same number of units in each hidden layer.
9. Training a neural network
	1. Randomly initialize the weights
	2. Implement forward propagation to get h(Theta) for all x(i)
	3. Implement the cost function J(Theta)
	4. Implement backpropagation to compute partial derivatives
	5. Use gradient approximation to confirm backpropagation works, disable gradient approximation
	6. Use gradient descent or an optimization function to minimize the cost function with the weights in theta.
	7. Preform forward and backwards propagation on m training examples
(Keep in mind J(Theta) is not convex and can end up in a local minimum instead of a global minimum)

