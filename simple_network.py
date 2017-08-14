# Implementa una red simple,
# utilizando la funcion sigmoid como funcion de activacion
# J. I. Vasquez

import numpy as np

def sigmoid(x):
    sg = 1/(1+np.exp(-x))
    return sg

def function_h(X, W, b):
    sum = 0
    Mult = np.multiply(W,X)
    for p in Mult:
        sum = sum + p
    sum = sum + b
    return sum

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

h = function_h(inputs,weights,bias)
output = sigmoid(h)

print('Output:')
print(output)
