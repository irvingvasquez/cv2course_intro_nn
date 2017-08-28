import numpy as np

# Impleementa un paso del gradiente descendiente
# J Irving Vasquez

# activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# f prime
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# function h
def function_h(X, W, b):
    return np.dot(W, X) + b

# output of the NN
def function_f(X,W,b):
    return sigmoid(function_h(X,W,b))

# error term, delta
def error_term(y,W,X):
    error = y - function_f(X,W,0)
    return error * sigmoid_prime(function_h(X,W,0))

# weight updates
def weight_update(W, X, eta, i, y):
    return w[i] + eta * error_term(y,W,X) * X[i] 

learnrate = 0.4
x = np.array([2, 1])
y = np.array(0.6)

# Initial weights
w = np.array([-0.5, 0.5])

# Calculate one gradient descent step for each weight
# TODO: Calculate output of neural network
nn_output = function_f(x,w,0)

# TODO: Calculate error of neural network
error = y - function_f(x,w,0)

# TODO: Calculate change in weights
del_w = [learnrate * error_term(y,w,x) * x[0], learnrate * error_term(y,w,x) * x[1]]

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
