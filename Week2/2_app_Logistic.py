#Logistic Regression with a Neural Network mindset
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *
######2 - Overview of the Problem set
####      Problem Statement: You are given a dataset ("data.h5") containing:
##- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)

##- a test set of m_test images labeled as cat or non-cat

##- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
##  Thus, each image is square (height = num_px) and (width = num_px)

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# We added "_orig" at the end of image datasets (train and test) because we are going
# to preprocess them. After preprocessing, we will end up with train_set_x and
# test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).

# Each line of your train_set_x_orig and test_set_x_orig is an array representing an image.
# You can visualize an example by running the following code. Feel free also to change the
# index value and re-run to see other images.

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
plt.show()
# Many software bugs in deep learning come from having matrix/vector dimensions that don't
# fit. If you can keep your matrix/vector dimensions straight you will go a long way toward
# eliminating many bugs.


# Find the values for:

# - m_train (number of training examples)
# - m_test (number of test examples)
# - num_px (= height = width of a training image)

#Remember that train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3).
# For instance, you can access m_train by writing train_set_x_orig.shape[0]


#(??? 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[2]


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# For convenience, you should now reshape images of shape (num_px, num_px, 3) in a
# numpy-array of shape (num_px  ???  num_px  ???  3, 1). After this, our training (and test)
# dataset is a numpy-array where each column represents a flattened image. There should
# be m_train (respectively m_test) columns.

# A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of
# shape (b ??? c ??? d, a) is to use:

# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# To represent color images, the red, green and blue channels (RGB) must be specified for
# each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.

# One common preprocessing step in machine learning is to center and standardize your
# dataset, meaning that you substract the mean of the whole numpy array from each example,
# and then divide each example by the standard deviation of the whole numpy array.
# But for picture datasets, it is simpler and more convenient and works almost as well
# to just divide every row of the dataset by 255 (the maximum value of a pixel channel).

# Let's standardize our dataset.
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# What you need to remember:

# Common steps for pre-processing a new dataset are:

# Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
# Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
# "Standardize" the data


#### Mathematical expression of the algorithm:

#For one example ????(????):
#????(????)=????.???? ????(????)+????
#?????? (????)=????(????)=????????????????????????????(????(????))
#L(????(????),????(????))=???????(????)log(????(????))???(1???????(????))log(1???????(????))
#The cost is then computed by summing over all training examples:
#J=(1/????)???????=1 to ????   L(????(????),????(????))

#### The main steps for building a Neural Network are:

#    1- Define the model structure (such as number of input features)
#    2- Initialize the model's parameters
#    3- Loop:
#            - Calculate current loss (forward propagation)
#            - Calculate current gradient (backward propagation)
#            - Update parameters (gradient descent)
# You often build 1-3 separately and integrate them into one function we call model()

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1/(1+np.exp(-z))
    return s


# GRADED FUNCTION: initialize_with_zeros
Z=np.array([0,2])
sigmoid(z)


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """
    #w=np.zeros(dim).reshape(dim,1)
    w=np.array([0]*dim).reshape(dim,1)
    b=0.
    return w, b

dim=2
initialize_with_zeros(dim)


# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    A = sigmoid(np.dot(w.T,X)+b)
    # compute cost by using np.dot to perform multiplication.
    # And don't use loops for the sum.
    cost=(-1/m)*sum(sum(((Y*np.log(A))+(((1-Y)*np.log(1-A))))))
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw =(1/m)*np.dot(X,(A-Y).T)
    db=1/m*sum(sum(A-Y))
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    return grads, cost



w =  np.array([[1.], [2]])
b = 1.5
X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
Y = np.array([[1, 1, 0]])
grads, cost = propagate(w, b, X, Y)


#### Optimization
#Write down the optimization function. The goal is to learn  ????  and  ????  by minimizing the
# cost function  ???? . For a parameter  ???? , the update rule is  ????=??????????? ???????? , where  ????  is the
# learning rate.

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    #
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    #
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    #
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w,b,X,Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule
        w=w-learning_rate*dw
        b=b-learning_rate*db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

#### predict
# - Calculate  ?????? =????=????(????????????+????)
# - Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5),
# stores the predictions in a vector Y_prediction. If you wish, you can use an if/else
# statement in a for loop (though there is also a way to vectorize this).

# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    #
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    #
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    #
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A =sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
         if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
         else:
            Y_prediction[0,i] = 0
    return Y_prediction

w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
predict(w, b, X)
#### What to remember:
## You've implemented several functions that:
## Initialize (w,b)
##Optimize the loss iteratively to learn parameters (w,b):
##      Computing the cost and its gradient
##      Updating the parameters using gradient descent
##Use the learned (w,b) to predict the labels for a given set of examples


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros
    w,b=initialize_with_zeros(X_train.shape[0])
    # Gradient descent
    # params, grads, costs = ...
    params, grads, costs=optimize(w, b, X_train, Y_train, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=False)
    # Retrieve parameters w and b from dictionary "params"
    w=params['w']
    b=params['b']
    # Predict test/train set examples
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

# train your model
dim=X.shape[0]
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Example of a picture that was wrongly classified.
index = 8
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")
plt.show()

#Let's also plot the cost function and the gradients.
# Plot learning curve (with costs)
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()
#Interpretation: You can see the cost decreasing. It shows that the parameters are being
# learned. However, you see that you could train the model even more on the training set.
# Try to increase the number of iterations in the cell above and rerun the cells. You might
# see that the training set accuracy goes up, but the test set accuracy goes down. This is
# called overfitting.




#Choice of learning rate
#Reminder: In order for Gradient Descent to work you must choose the learning rate wisely.
# The learning rate  ????  determines how rapidly we update the parameters. If the learning rate
# is too large we may "overshoot" the optimal value. Similarly, if it is too small we will
# need too many iterations to converge to the best values. That's why it is crucial to use
# a well-tuned learning rate.

#Let's compare the learning curve of our model with several choices of learning rates. Run
# the cell below. This should take about 1 minute. Feel free also to try different values
# than the three we have initialized the learning_rates variable to contain, and see what
# happens.
learning_rates = [0.01, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print ("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()



#Interpretation:
#Different learning rates give different costs and thus different predictions results.
#If the learning rate is too large (0.01), the cost may oscillate up and down. It may even
# diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost).
#A lower cost doesn't mean a better model. You have to check if there is possibly overfitting.
# It happens when the training accuracy is a lot higher than the test accuracy.
#In deep learning, we usually recommend that you:
#Choose the learning rate that better minimizes the cost function.
#If your model overfits, use other techniques to reduce overfitting.



# change this to the name of your image file
my_image = "my_image.jpg"

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


#What to remember from this assignment:

#Preprocessing the dataset is important.
#You implemented each function separately: initialize(), propagate(), optimize().
# Then you built a model().
#Tuning the learning rate (which is an example of a "hyperparameter") can make a big
# difference to the algorithm. You will see more examples of this later in this course!
