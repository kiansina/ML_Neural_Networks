###### Planar data classification with one hidden layer:
# Package imports
import numpy as np
import copy
import matplotlib.pyplot as plt
from testCases_v2 import *
from public_tests import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()
# he data looks like a "flower" with some red (label y=0) and some blue (y=1) points.
# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()
#You have:
#- a numpy-array (matrix) X that contains your features (x1, x2)
#- a numpy-array (vector) Y that contains your labels (red:0, blue:1)


shape_X = X.shape
shape_Y = Y.shape
# training set size
m=len(X[0,:])

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

#####3 - Simple Logistic Regression
#Before building a full neural network, let's check how logistic regression performs on this
#problem. You can use sklearn's built-in functions for this. Run the code below to train a
# logistic regression classifier on the dataset.
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

#You can now plot the decision boundary of these models!
# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

#Interpretation: The dataset is not linearly separable, so logistic regression doesn't
# perform well. Hopefully a neural network will do better. Let's try this now!


#### Interpretation: The dataset is not linearly separable, so logistic regression doesn't
#     perform well. Hopefully a neural network will do better. Let's try this now!


#Reminder: The general methodology to build a Neural Network is to:
#1. Define the neural network structure ( # of input units,  # of hidden units, etc).
#2. Initialize the model's parameters
#3. Loop:
#    - Implement forward propagation
#    - Compute loss
#    - Implement backward propagation to get the gradients
#    - Update parameters (gradient descent)
#In practice, you'll often build helper functions to compute steps 1-3, then merge them into
# one function called nn_model(). Once you've built nn_model() and learned the right parameters,
# you can make predictions on new data.



#Defining the neural network structure
#Define three variables:
#- n_x: the size of the input layer
#- n_h: the size of the hidden layer (**set this to 4, only for this Exercise 2**)
#- n_y: the size of the output layer
#Hint: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.

# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

#________________test function
t_X, t_Y = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(t_X, t_Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

# Initialize the model's parameters
# You will initialize the weights matrices with random values.
#     Use: np.random.randn(a,b) * 0.01 to randomly initialize a matrix of shape (a,b).
# You will initialize the bias vectors as zeros.
#     Use: np.zeros((a,b)) to initialize a matrix of shape (a,b) with zeros.
# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    #
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    #
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    #
    return parameters

#________________test function
np.random.seed(2)
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))




#Implement forward_propagation() using the following equations:

#𝑍[1]=𝑊[1]𝑋+𝑏[1]
#𝐴[1]=tanh(𝑍[1])
#𝑍[2]=𝑊[2]𝐴[1]+𝑏[2]
#𝑌̂ =𝐴[2]=𝜎(𝑍[2])

#Instructions:
#Values needed in the backpropagation are stored in "cache". The cache will be given as an
#input to the backpropagation function.


# GRADED FUNCTION:forward_propagation
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    #
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    #
    assert(A2.shape == (1, X.shape[1]))
    #
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    #
    return A2, cache

#________________test function
t_X, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(t_X, parameters)
print("A2 = " + str(A2))

forward_propagation_test(forward_propagation)

#### Compute the Cost (check photo 2 in the folder to see the formula)
# GRADED FUNCTION: compute_cost
def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost given in equation (13)
    #
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    #
    Returns:
    cost -- cross-entropy cost given equation (13)
    #
    """
    m = Y.shape[1] # number of examples
    # Compute the cross-entropy cost
    logprobs = Y*np.log(A2)+(1-Y)*np.log(1-A2)
    cost = (-1/m)*np.sum(logprobs)

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
                                    # E.g., turns [[17]] into 17

    return cost


#________________test function
A2, t_Y = compute_cost_test_case()
cost = compute_cost(A2, t_Y)
print("cost = " + str(compute_cost(A2, t_Y)))




#### Implement Backpropagation (photo 3 to chech formula)
#Using the cache computed during forward propagation, you can now implement backward propagation.
# GRADED FUNCTION: backward_propagation
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))
    dW1 = (1/m)*(np.dot(dZ1,X.T))
    db1 = (1/m)*(np.sum(dZ1,axis=1,keepdims=True))
    #
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    #
    return grads

#________________test function
parameters, cache, t_X, t_Y = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, t_X, t_Y)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))



#### Update Parameters
#Implement the update rule. Use gradient descent. You have to use (dW1, db1, dW2, db2) in
# order to update (W1, b1, W2, b2).

#General gradient descent rule:  𝜃=𝜃−𝛼∂𝐽∂𝜃  where  𝛼  is the learning rate and  𝜃  represents a parameter.

#Hint
#Use copy.deepcopy(...) when copying lists or dictionaries that are passed as parameters to
# functions. It avoids input parameters being modified within the function. In some scenarios,
# this could be inefficient, but it is required for grading purposes.
# GRADED FUNCTION: update_parameters
def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve a copy of each parameter from the dictionary "parameters". Use copy.deepcopy(...) for W1 and W2
    W1 = copy.deepcopy(parameters['W1'])
    b1 = copy.deepcopy(parameters['b1'])
    W2 = copy.deepcopy(parameters['W2'])
    b2 = copy.deepcopy(parameters['b2'])
    # Retrieve each gradient from the dictionary "grads"
    dW1 = copy.deepcopy(grads['dW1'])
    db1 = copy.deepcopy(grads['db1'])
    dW2 = copy.deepcopy(grads['dW2'])
    db2 = copy.deepcopy(grads['db2'])
    # Update rule for each parameter
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    #
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    #
    return parameters

#________________test function
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

update_parameters_test(update_parameters)


#### Integration
#Integrate your functions in nn_model()
# GRADED FUNCTION: nn_model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        # Cost function. Inputs: "A2, Y". Outputs: "cost".
        cost = compute_cost(A2, Y)
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    #
    return parameters


#________________test function
nn_model_test(nn_model)





#### Test the Model
# Predict
# GRADED FUNCTION: predict
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions =  (A2 >= 0.5)
    return predictions

#________________test function
parameters, t_X = predict_test_case()

predictions = predict(parameters, t_X)
print("Predictions: " + str(predictions))


#### Test the Model on the Planar Dataset
#It's time to run the model and see how it performs on a planar dataset. Run the following
# code to test your model with a single hidden layer of  𝑛ℎ  hidden units!

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

#Accuracy is really high compared to Logistic Regression. The model has learned the patterns
# of the flower's petals! Unlike logistic regression, neural networks are able to learn even
# highly non-linear decision boundaries.


#### Tuning hidden layer size
#Run the following code(it may take 1-2 minutes). Then, observe different behaviors of the
# model for various hidden layer sizes.
# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

plt.show()
#Interpretation:
#The larger models (with more hidden units) are able to fit the training set better, until
# eventually the largest models overfit the data.

# The best hidden layer size seems to be around n_h = 5. Indeed, a value around here seems to
# fits the data well without also incurring noticeable overfitting.

#Later, you'll become familiar with regularization, which lets you use very large models
# (such as n_h = 50) without much overfitting.




#### Performance on other datasets
# If you want, you can rerun the whole notebook (minus the dataset part) for each of the
# following datasets.
# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
