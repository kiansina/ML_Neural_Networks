#Deep Neural Network for Image Classification: Application
#To build your cat/not-a-cat classifier, you'll use the functions from the previous
#    assignment to build a deep network. Hopefully, you'll see an improvement in accuracy
#    over your previous logistic regression implementation

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *
from public_tests import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


#Load and Process the Dataset
#You'll be using the same "Cat vs non-Cat" dataset as in "Logistic Regression as a Neural
#     Network" (Assignment 2). The model you built back then had 70% test accuracy on
#     classifying cat vs non-cat images. Hopefully, your new model will perform even better!

#Problem Statement: You are given a dataset ("data.h5") containing:

#   - a training set of `m_train` images labelled as cat (1) or non-cat (0)
#   - a test set of `m_test` images labelled as cat and non-cat
#   - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# The following code will show you an image in the dataset. Feel free to change the index
#    and re-run the cell multiple times to check out other images.

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
plt.show()



# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))



#As usual, you reshape and standardize the images before feeding them to the network.
#    The code is given in the cell below.

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

#Note:  12,288  equals  64×64×3 , which is the size of one reshaped image vector.

#### Model Architecture
#2-layer Neural Network
#You're going to build two different models:

#A 2-layer neural network
#An L-layer deep neural network
#Then, you'll compare the performance of these models, and try out some different values for  𝐿 .


####2-layer neural network.
#The model can be summarized as: INPUT -> LINEAR -> RELU -> LINEAR -> SIGMOID -> OUTPUT


#Detailed Architecture of Figure 2:

#The input is a (64,64,3) image which is flattened to a vector of size  (12288,1) .
#The corresponding vector:  [𝑥0,𝑥1,...,𝑥12287]𝑇  is then multiplied by the weight matrix  𝑊[1]  of size  (𝑛[1],12288) .
#Then, add a bias term and take its relu to get the following vector:  [𝑎[1]0,𝑎[1]1,...,𝑎[1]𝑛[1]−1]𝑇 .
#Repeat the same process.
#Multiply the resulting vector by  𝑊[2]  and add the intercept (bias).
#Finally, take the sigmoid of the result. If it's greater than 0.5, classify it as a cat.


####L-layer Deep Neural Network
#L-layer neural network.
#The model can be summarized as: [LINEAR -> RELU]  ×  (L-1) -> LINEAR -> SIGMOID

#Detailed Architecture of Figure 3:

#The input is a (64,64,3) image which is flattened to a vector of size (12288,1).
#The corresponding vector:  [𝑥0,𝑥1,...,𝑥12287]𝑇  is then multiplied by the weight matrix  𝑊[1]
#     and then you add the intercept  𝑏[1] . The result is called the linear unit.

#Next, take the relu of the linear unit. This process could be repeated several times for each  (𝑊[𝑙],𝑏[𝑙])
#     depending on the model architecture.

#Finally, take the sigmoid of the final linear unit. If it is greater than 0.5, classify it as a cat.


####General Methodology
#As usual, you'll follow the Deep Learning methodology to build the model:

#    Initialize parameters / Define hyperparameters
#    Loop for num_iterations: a. Forward propagation b. Compute cost function
#         c. Backward propagation d. Update parameters (using parameters, and grads from backprop)
#    Use trained parameters to predict labels

#Now go ahead and implement those two models!



####Two-layer Neural Network

#Use the helper functions you have implemented in the previous assignment to build a 2-layer
#    neural network with the following structure: LINEAR -> RELU -> LINEAR -> SIGMOID.
#    The functions and their inputs are:

#def initialize_parameters(n_x, n_h, n_y):
#    ...
#    return parameters
#def linear_activation_forward(A_prev, W, b, activation):
#    ...
#    return A, cache
#def compute_cost(AL, Y):
#    ...
#    return cost
#def linear_activation_backward(dA, cache, activation):
#    ...
#    return dA_prev, dW, db
#def update_parameters(parameters, grads, learning_rate):
#    ...
#    return parameters

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075

# GRADED FUNCTION: two_layer_model

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    #
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations
    #
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    #
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    #
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        # Compute cost
        cost = compute_cost(A2, Y)
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
        #
    return parameters, costs





def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()



#______________________________________________test function______________________________
parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2, print_cost=False)

print("Cost after first iteration: " + str(costs[0]))

two_layer_model_test(two_layer_model)
##########################################################################################

####Train the model
#If your code passed the previous cell, run the cell below to train your parameters.

#The cost should decrease on every iteration.

#It may take up to 5 minutes to run 2500 iterations.

parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
plot_costs(costs, learning_rate)

#Nice! You successfully trained the model. Good thing you built a vectorized implementation!
#    Otherwise it might have taken 10 times longer to train this.

#Now, you can use the trained parameters to classify images from the dataset. To see your
#    predictions on the training and test sets, run the cell below.

predictions_train = predict(train_x, train_y, parameters)

predictions_test = predict(test_x, test_y, parameters)

#Congratulations! It seems that your 2-layer neural network has better performance (72%)
#    than the logistic regression implementation (70%, assignment week 2). Let's see if you can
#    do even better with an  𝐿 -layer model.


#Note: You may notice that running the model on fewer iterations (say 1500) gives better
#    accuracy on the test set. This is called "early stopping" and you'll hear more about
#    it in the next course. Early stopping is a way to prevent overfitting.



####L-layer Neural Network


#Use the helper functions you implemented previously to build an  𝐿 -layer neural network
#    with the following structure: *[LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID*. The
#    functions and their inputs are:

#def initialize_parameters_deep(layers_dims):
#    ...
#    return parameters
#def L_model_forward(X, parameters):
#    ...
#    return AL, caches
#def compute_cost(AL, Y):
#    ...
#    return cost
#def L_model_backward(AL, Y, caches):
#    ...
#    return grads
#def update_parameters(parameters, grads, learning_rate):
#    ...
#    return parameters

### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)
    costs = []                         # keep track of cost
    #
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    #
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        #
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        #
        # Compute cost.
        cost = compute_cost(AL, Y)
        # Backward propagation.
        grads =L_model_backward(AL, Y, caches)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    #
    return parameters, costs


#______________________________________________test function______________________________
parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1, print_cost = False)

print("Cost after first iteration: " + str(costs[0]))

L_layer_model_test(L_layer_model)
##########################################################################################

#Train the model

#If your code passed the previous cell, run the cell below to train your model as a
#    4-layer neural network.

# The cost should decrease on every iteration.

# It may take up to 5 minutes to run 2500 iterations.

parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)

#It seems that your 4-layer neural network has better performance (80%) than your 2-layer
#     neural network (72%) on the same test set.

#This is pretty good performance for this task. Nice job!

#In the next course on "Improving deep neural networks," you'll be able to obtain even
#    higher accuracy by systematically searching for better hyperparameters: learning_rate,
#    layers_dims, or num_iterations, for example.

#Results Analysis
#First, take a look at some images the L-layer model labeled incorrectly. This will show a
#     few mislabeled images.
print_mislabeled_images(classes, test_x, test_y, pred_test)
plt.show()

#A few types of images the model tends to do poorly on include:
#    Cat body in an unusual position
#    Cat appears against a background of a similar color
#    Unusual cat color and species
#    Camera Angle
#    Brightness of the picture
#    Scale variation (cat is very large or small in image)



####Test with your own image (optional/ungraded exercise)
#From this point, if you so choose, you can use your own image to test the output of your
#     model. To do that follow these steps:

#    Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
#    Add your image to this Jupyter Notebook's directory, in the "images" folder
#    Change your image's name in the following code
#    Run the code and check if the algorithm is right (1 = cat, 0 = non-cat)!



my_image = "my_image.jpg" # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)


fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T

my_predicted_image = predict(image, my_label_y, parameters)


print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
