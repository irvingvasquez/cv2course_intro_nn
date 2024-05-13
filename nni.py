
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

def beale_function(x, y):
    term1 = (1.5 - x + x * y)**2
    term2 = (2.25 - x + x * y**2)**2
    term3 = (2.625 - x + x * y**3)**2
    return term1 + term2 + term3

def beale_gradient_x(x, y):
    return 2 * (1.5 - x + x * y) * (-1 + y) + 2 * (2.25 - x + x * y**2) * (-1 + y**2) + 2 * (2.625 - x + x * y**3) * (-1 + y**3)

def beale_gradient_y(x, y):
    return 2 * (1.5 - x + x * y) * x + 2 * (2.25 - x + x * y**2) * 2 * x * y + 2 * (2.625 - x + x * y**3) * 3 * x * y**2


