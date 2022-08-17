import numpy as np
import time
import math

a=np.random.rand(10000000)
b=np.random.rand(10000000)

tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(toc-tic)

d=0
tic=time.time()
for i in range(10000000):
    d+=a[i]*b[i]

toc=time.time()
print(toc-tic)


## Another:
v=np.random.rand(10000000,1)
tic=time.time()
u=np.zeros((10000000,1))
for i in range(10000000):
    u[i]=math.exp(v[i])

toc=time.time()
print(toc-tic)

tic=time.time()
h=np.exp(v)
toc=time.time()
print(toc-tic)

## Another:
np.log(u)
np.abs(u)
np.maximum(v,0)
v**2
1/v

# Broadcasting :
A=np.array([[56, 0, 4.4, 68],
            [1.2, 104, 52, 8],
            [1.8, 135, 99, 0.9]])

print(A)

cal=A.sum(axis=0)
print(cal)

percentage=100*A/cal.reshape(1,4)
print(percentage)

## Extra info on numpy:
np.exp? #in jupyter if you write that, it will open extra documentation


#################################################################################
## Normalizing
#https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
X=np.array([[0,3,4],[2,6,4]])
x_norm=np.linalg.norm(X,ord=2, axis=1, keepdims=True) #This is normalising when ord is 2 means : the square root of the sum of each component squared
#When it is 1 it is just sum of them (radical ba rishe 0 om)
X2=X/x_norm




#################################################################################
## Softmax
#https://www.coursera.org/learn/neural-networks-deep-learning/programming/isoAV/python-basics-with-numpy/lab?path=%2Fnotebooks%2Frelease%2FW2A1%2FPython_Basics_with_Numpy.ipynb
X=np.array([[0,3,4],[2,6,4]])
x_exp=np.exp(X)
x_sum=np.sum(x_exp,axis=1,keepdims=True)
s=x_exp/x_sum
#sigmoid(x)=1/(1+exp(-x))
#                          : s=1/(1+np.exp(-x))
#sigmoid_derivatives: sigmoid(x)*(1-sigmoid(x))
#                          : ds=(1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))




##################################################################################
## Vectorization:
x1=X
x2=X+3

### VECTORIZED DOT PRODUCT OF VECTORS ###
dot = np.dot(x1,x2)
### VECTORIZED OUTER PRODUCT ###
outer = np.outer(x1,x2)
### VECTORIZED ELEMENTWISE MULTIPLICATION ###
mul = np.multiply(x1,x2)
#or
mul = x1*x2
### VECTORIZED GENERAL DOT PRODUCT ###
W = np.random.rand(3,len(x1))
dot = np.dot(W,x1)


#################################################################################
## Loss functions: L1
#if y is : vector of size m (true labels)
#if yhat is : vector of size m (predicted labels)
loss=sum(abs(y-yhat))
## Loss functions: L2
loss=np.dot((y-yhat),(y-yhat))
