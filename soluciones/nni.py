# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np
import math

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))
    
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def function_and_gradients(eq):
    if eq == 1:
        f = lambda x, y: 0.1 * x**2 + 2 * y**2
        dfx = lambda x, y: 0.2 * x
        dfy = lambda x, y: 4 * y
    elif eq == 2:
        f = lambda x, y: math.cos(x) + math.sin(y)
        dfx = lambda x, y: -math.sin(x)
        dfy = lambda x, y: math.cos(y)
    elif eq == 3:
        a = 1
        b = 100
        f = lambda x, y: (a - x)**2 + b * (y - x**2)**2
        dfx = lambda x, y: -2 * (a - x) - 4 * b * x * (y - x**2)
        dfy = lambda x, y: 2 * b * (y - x**2)
    else:
        raise ValueError("Invalid equation number.")

    return f, dfx, dfy