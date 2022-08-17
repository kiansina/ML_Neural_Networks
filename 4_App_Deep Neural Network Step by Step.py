#Building your Deep Neural Network: Step by Step

#Notation:
#Superscript  [𝑙]  denotes a quantity associated with the  𝑙𝑡ℎ  layer.
#Example:  𝑎[𝐿]  is the  𝐿𝑡ℎ  layer activation.  𝑊[𝐿]  and  𝑏[𝐿]  are the  𝐿𝑡ℎ  layer parameters.
#Superscript  (𝑖)  denotes a quantity associated with the  𝑖𝑡ℎ  example.
#Example:  𝑥(𝑖)  is the  𝑖𝑡ℎ  training example.
#Lowerscript  𝑖  denotes the  𝑖𝑡ℎ  entry of a vector.
#Example:  𝑎[𝑙]𝑖  denotes the  𝑖𝑡ℎ  entry of the  𝑙𝑡ℎ  layer's activations).


import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from public_tests import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



np.random.seed(1)


#### Initialization:
#The model's structure is: LINEAR -> RELU -> LINEAR -> SIGMOID.
#Use this random initialization for the weight matrices: np.random.randn(d0, d1, ..., dn) * 0.01 with
#    the correct shape. The documentation for np.random.randn
#Use zero initialization for the biases: np.zeros(shape). The documentation for np.zeros
# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(1)
    #
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros([n_h,1])
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros([n_y,1])
    #
    #
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    #
    return parameters

#______________________________________________test function______________________________
parameters = initialize_parameters(3,2,1)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_test(initialize_parameters)
##########################################################################################
#### initialize_parameters_deep:
#Instructions:

#The model's structure is *[LINEAR -> RELU]  ×  (L-1) -> LINEAR -> SIGMOID*. I.e., it has  𝐿−1
#    layers using a ReLU activation function followed by an output layer with a sigmoid activation function.

#Use random initialization for the weight matrices. Use np.random.randn(d0, d1, ..., dn) * 0.01.
#Use zeros initialization for the biases. Use np.zeros(shape).
#You'll store  𝑛[𝑙] , the number of units in different layers, in a variable layer_dims. For
#    example, the layer_dims for last week's Planar Data classification model would have been [2,4,1]:
#    There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit.
#    This means W1's shape was (4,2), b1 was (4,1), W2 was (1,4) and b2 was (1,1). Now you will
#    generalize this to  𝐿  layers!

#Here is the implementation for  𝐿=1  (one layer neural network). It should inspire you to implement the general case (L-layer neural network).
#  if L == 1:
#      parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
#      parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))

# GRADED FUNCTION: initialize_parameters_deep
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network
    #
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        #
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        #
    return parameters
#______________________________________________test function______________________________
parameters = initialize_parameters_deep([5,4,3])

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_deep_test(initialize_parameters_deep)
##########################################################################################
#### Forward Propagation Module:
#Now that you have initialized your parameters, you can do the forward propagation module.
# Start by implementing some basic functions that you can use again later when implementing the
# model. Now, you'll complete three functions in this order:
    #LINEAR
    #LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid.
    #[LINEAR -> RELU]  ×  (L-1) -> LINEAR -> SIGMOID (whole model)

#The linear forward module (vectorized over all the examples) computes the following equations:
#    𝑍[𝑙]=𝑊[𝑙] 𝐴[𝑙−1]+𝑏[𝑙]
#    where  𝐴[0]=𝑋

#Build the linear part of forward propagation.
#Reminder: The mathematical representation of this unit is  𝑍[𝑙]=𝑊[𝑙] 𝐴[𝑙−1] + 𝑏[𝑙]
#    You may also find np.dot() useful. If your dimensions don't match, printing W.shape may help.
# GRADED FUNCTION: linear_forward

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z, cache
#______________________________________________test function______________________________

t_A, t_W, t_b = linear_forward_test_case()
t_Z, t_linear_cache = linear_forward(t_A, t_W, t_b)
print("Z = " + str(t_Z))

linear_forward_test(linear_forward)
##########################################################################################
####Linear-Activation Forward
#In this notebook, you will use two activation functions:

#Sigmoid:  𝜎(𝑍)=𝜎(𝑊𝐴+𝑏)=1/(1+𝑒**−(𝑊𝐴+𝑏)) . You've been provided with the sigmoid function which
#     returns two items: the activation value "a" and a "cache" that contains "Z" (it's what we
#     will feed in to the corresponding backward function). To use it you could just call:
#     A, activation_cache = sigmoid(Z)

#ReLU: The mathematical formula for ReLu is  𝐴=𝑅𝐸𝐿𝑈(𝑍)=𝑚𝑎𝑥(0,𝑍) . You've been provided with the
#    relu function. This function returns two items: the activation value "A" and a "cache" that
#    contains "Z" (it's what you'll feed in to the corresponding backward function). To use it
#    you could just call:
#    A, activation_cache = relu(Z)

#For added convenience, you're going to group two functions (Linear and Activation) into one
#    function (LINEAR->ACTIVATION). Hence, you'll implement a function that does the LINEAR
#    forward step, followed by an ACTIVATION forward step.

#implement the forward propagation of the LINEAR->ACTIVATION layer. Mathematical relation is:
#    𝐴[𝑙]=𝑔(𝑍[𝑙])=𝑔(𝑊[𝑙]𝐴[𝑙−1]+𝑏[𝑙])  where the activation "g" can be sigmoid() or relu().
#    Use linear_forward() and the correct activation function.

# GRADED FUNCTION: linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    #
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        #
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        #
    cache = (linear_cache, activation_cache)
    #
    return A, cache

#______________________________________________test function______________________________
t_A_prev, t_W, t_b = linear_activation_forward_test_case()

t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "sigmoid")
print("With sigmoid: A = " + str(t_A))

t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "relu")
print("With ReLU: A = " + str(t_A))

linear_activation_forward_test(linear_activation_forward)
##########################################################################################
#Note: In deep learning, the "[LINEAR->ACTIVATION]" computation is counted as a single
#    layer in the neural network, not two layers

#### L-Layer Model
#For even more convenience when implementing the  𝐿 -layer Neural Net, you will need a function
#    that replicates the previous one (linear_activation_forward with RELU)  𝐿−1  times, then
#    follows that with one linear_activation_forward with SIGMOID.

#implement the forward propagation of the above model.
#Instructions: In the code below, the variable AL will denote  𝐴[𝐿]=𝜎(𝑍[𝐿])=𝜎(𝑊[𝐿]𝐴[𝐿−1]+𝑏[𝐿])
#    (This is sometimes also called Yhat, i.e., this is  𝑌̂  .)

#Hints:
#Use the functions you've previously written
#Use a for loop to replicate [LINEAR->RELU] (L-1) times
#Don't forget to keep track of the caches in the "caches" list. To add a new value c to a list,
#    you can use list.append(c).

# GRADED FUNCTION: L_model_forward

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    #
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A
        A, cache =linear_activation_forward(A_prev, parameters['W{}'.format(l)], parameters['b{}'.format(l)], "relu")
        caches.append(cache)
    #
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W{}'.format(L)], parameters['b{}'.format(L)], "sigmoid")
    caches.append(cache)
    #
    return AL, caches

#______________________________________________test function______________________________

t_X, t_parameters = L_model_forward_test_case_2hidden()
t_AL, t_caches = L_model_forward(t_X, t_parameters)

print("AL = " + str(t_AL))

L_model_forward_test(L_model_forward)
##########################################################################################
#Awesome! You've implemented a full forward propagation that takes the input X and outputs a
#    row vector  𝐴[𝐿]  containing your predictions. It also records all intermediate values in
#    "caches". Using  𝐴[𝐿] , you can compute the cost of your predictions.

#### Cost Function:
#Now you can implement forward and backward propagation! You need to compute the cost, in order
#    to check whether your model is actually learning.

#compute_cost:
#Compute the cross-entropy cost  𝐽 , using the following formula: (check photo 2)
#     −1/𝑚∑(for 𝑖=1 to 𝑚) : (𝑦(𝑖)log(𝑎[𝐿](𝑖))+(1−𝑦(𝑖))log(1−𝑎[𝐿](𝑖)))



# GRADED FUNCTION: compute_cost
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).
    #
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    #
    Returns:
    cost -- cross-entropy cost
    """
    #
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = (-1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL),axis=1)
    #
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    #
    return cost

#______________________________________________test function______________________________
t_Y, t_AL = compute_cost_test_case()
t_cost = compute_cost(t_AL, t_Y)

print("Cost: " + str(t_cost))

compute_cost_test(compute_cost)
##########################################################################################

#### Backward Propagation Module:

#Just as you did for the forward propagation, you'll implement helper functions for backpropagation.
#    Remember that backpropagation is used to calculate the gradient of the loss function
#    with respect to the parameters.


#Now, similarly to forward propagation, you're going to build the backward propagation in three steps:
#    LINEAR backward
#    LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
#    [LINEAR -> RELU]  ×  (L-1) -> LINEAR -> SIGMOID backward (whole model)

#For the next exercise, you will need to remember that:

#b is a matrix(np.ndarray) with 1 column and n rows, i.e: b = [[1.0], [2.0]] (remember that b is a constant)

#np.sum performs a sum over the elements of a ndarray
#axis=1 or axis=0 specify if the sum is carried out by rows or by columns respectively
#keepdims specifies if the original dimensions of the matrix must be kept.
#Look at the following example to clarify:

A = np.array([[1, 2], [3, 4]])

print('axis=1 and keepdims=True')
print(np.sum(A, axis=1, keepdims=True))
print('axis=1 and keepdims=False')
print(np.sum(A, axis=1, keepdims=False))
print('axis=0 and keepdims=True')
print(np.sum(A, axis=0, keepdims=True))
print('axis=0 and keepdims=False')
print(np.sum(A, axis=0, keepdims=False))

#### Linear Backward:
#For layer  𝑙 , the linear part is:  𝑍[𝑙]=𝑊[𝑙]𝐴[𝑙−1]+𝑏[𝑙]  (followed by an activation).

#Suppose you have already calculated the derivative  𝑑𝑍[𝑙]=∂L∂𝑍[𝑙] . You want to get  (𝑑𝑊[𝑙],𝑑𝑏[𝑙],𝑑𝐴[𝑙−1]) .
#The three outputs  (𝑑𝑊[𝑙],𝑑𝑏[𝑙],𝑑𝐴[𝑙−1])  are computed using the input  𝑑𝑍[𝑙] .

#Here are the formulas you need: (photo 3):
#𝑑𝑊[𝑙]=∂L/∂𝑊[𝑙]=1/𝑚 𝑑𝑍[𝑙]𝐴[𝑙−1]𝑇
#𝑑𝑏[𝑙]=∂L/∂𝑏[𝑙]=1/𝑚 ∑(for 𝑖=1 to 𝑚) 𝑑𝑍[𝑙](𝑖)
#𝑑𝐴[𝑙−1]=∂L/∂𝐴[𝑙−1]=𝑊[𝑙]𝑇 𝑑𝑍[𝑙]

#𝐴[𝑙−1]𝑇  is the transpose of  𝐴[𝑙−1] .


#linear_backward
#Use the 3 formulas above to implement linear_backward().
#Hint:

#In numpy you can get the transpose of an ndarray A using A.T or A.transpose()
# GRADED FUNCTION: linear_backward

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)
    #
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    #
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    #
    dW = (1/m)*(np.dot(dZ,A_prev.T))
    db = (1/m)*(np.sum(dZ,axis=1,keepdims=True)) #sum by the rows of dZ with keepdims=True
    dA_prev = np.dot(W.T,dZ)
    #
    return dA_prev, dW, db

#______________________________________________test function______________________________
t_dZ, t_linear_cache = linear_backward_test_case()
t_dA_prev, t_dW, t_db = linear_backward(t_dZ, t_linear_cache)

print("dA_prev: " + str(t_dA_prev))
print("dW: " + str(t_dW))
print("db: " + str(t_db))

linear_backward_test(linear_backward)
##########################################################################################


#Linear-Activation Backward
#Next, you will create a function that merges the two helper functions: linear_backward
#    and the backward step for the activation linear_activation_backward.

#To help you implement linear_activation_backward, two backward functions have been provided:

#sigmoid_backward: Implements the backward propagation for SIGMOID unit. You can call it as follows:
#    dZ = sigmoid_backward(dA, activation_cache)

#relu_backward: Implements the backward propagation for RELU unit. You can call it as follows:
#    dZ = relu_backward(dA, activation_cache)

#If  𝑔(.)  is the activation function, sigmoid_backward and relu_backward compute
#    𝑑𝑍[𝑙]=𝑑𝐴[𝑙]∗𝑔′(𝑍[𝑙])

#linear_activation_backward
#Implement the backpropagation for the LINEAR->ACTIVATION layer.

# GRADED FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    #
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    #
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    #
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache)
    #
    elif activation == "sigmoid":
        dZ =  sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache)
    #
    return dA_prev, dW, db

#______________________________________________test function______________________________
t_dAL, t_linear_activation_cache = linear_activation_backward_test_case()

t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation = "sigmoid")
print("With sigmoid: dA_prev = " + str(t_dA_prev))
print("With sigmoid: dW = " + str(t_dW))
print("With sigmoid: db = " + str(t_db))

t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation = "relu")
print("With relu: dA_prev = " + str(t_dA_prev))
print("With relu: dW = " + str(t_dW))
print("With relu: db = " + str(t_db))

linear_activation_backward_test(linear_activation_backward)
##########################################################################################

#### L-Model Backward
# Now you will implement the backward function for the whole network!

# Recall that when you implemented the L_model_forward function, at each iteration, you
#    stored a cache which contains (X,W,b, and z). In the back propagation module, you'll
#    use those variables to compute the gradients. Therefore, in the L_model_backward function,
#    you'll iterate through all the hidden layers backward, starting from layer  𝐿 . On each step,
#    you will use the cached values for layer  𝑙  to backpropagate through layer  𝑙 . Figure 5 below
#    shows the backward pass


#### Initializing backpropagation:

#To backpropagate through this network, you know that the output is:  𝐴[𝐿]=𝜎(𝑍[𝐿]) Your code
#    thus needs to compute dAL  =∂L/∂𝐴[𝐿] . To do so, use this formula (derived using calculus
#    which, again, you don't need in-depth knowledge of!):
#   dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL

#You can then use this post-activation gradient dAL to keep going backward. As seen in Figure 5,
#    you can now feed in dAL into the LINEAR->SIGMOID backward function you implemented
#    (which will use the cached values stored by the L_model_forward function).

#After that, you will have to use a for loop to iterate through all the other layers using
#    the LINEAR->RELU backward function. You should store each dA, dW, and db in the grads
#    dictionary. To do so, use this formula :
#    𝑔𝑟𝑎𝑑𝑠["𝑑𝑊"+𝑠𝑡𝑟(𝑙)]=𝑑𝑊[𝑙]
#    For example, for  𝑙=3  this would store  𝑑𝑊[𝑙]  in grads["dW3"].


#### L_model_backward
#Implement backpropagation for the *[LINEAR->RELU]  ×  (L-1) -> LINEAR -> SIGMOID* model.

# GRADED FUNCTION: L_model_backward
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    #
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    #
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l-L]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    #
    return grads

#______________________________________________test function______________________________
t_AL, t_Y_assess, t_caches = L_model_backward_test_case()
grads = L_model_backward(t_AL, t_Y_assess, t_caches)

print("dA0 = " + str(grads['dA0']))
print("dA1 = " + str(grads['dA1']))
print("dW1 = " + str(grads['dW1']))
print("dW2 = " + str(grads['dW2']))
print("db1 = " + str(grads['db1']))
print("db2 = " + str(grads['db2']))

L_model_backward_test(L_model_backward)
##########################################################################################

####  Update Parameters:
#In this section, you'll update the parameters of the model, using gradient descent:

#   𝑊[𝑙]=𝑊[𝑙]−𝛼 𝑑𝑊[𝑙]
#   𝑏[𝑙]=𝑏[𝑙]−𝛼 𝑑𝑏[𝑙]

#where  𝛼  is the learning rate.

# After computing the updated parameters, store them in the parameters dictionary.
#Implement update_parameters() to update your parameters using gradient descent.

#Instructions: Update parameters using gradient descent on every  𝑊[𝑙]  and  𝑏[𝑙]  for  𝑙=1,2,...,𝐿 .
# GRADED FUNCTION: update_parameters
def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    #
    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward
    #
    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network
    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l + 1)]
    #
    return parameters

#______________________________________________test function______________________________
t_parameters, grads = update_parameters_test_case()
t_parameters = update_parameters(t_parameters, grads, 0.1)

print ("W1 = "+ str(t_parameters["W1"]))
print ("b1 = "+ str(t_parameters["b1"]))
print ("W2 = "+ str(t_parameters["W2"]))
print ("b2 = "+ str(t_parameters["b2"]))

update_parameters_test(update_parameters)
##########################################################################################


#Congratulations!
