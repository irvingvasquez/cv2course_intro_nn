
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


class Optimizer:
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance
        self.history = []

class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        super().__init__(tolerance=tolerance)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def next_params(self, gradient, params):
        return [param - self.learning_rate * grad for param, grad in zip(params, gradient)]
    
    def optimize(self, function, gradient, initial_params):
        self.history = []
        params = initial_params
        iteration = 0

        while iteration < self.max_iterations:
            log = {'iter': iteration, 'params': params, 'value': function(*params)}
            self.history.append(log)

            gradient_values = gradient(*params)
            updated_params = self.next_params(gradient_values, params)
            if self._converged(params, updated_params):
                break
            params = updated_params
            iteration += 1
        return params
    
    def _converged(self, params, updated_params):
        return all(abs(param - updated_param) < self.tolerance for param, updated_param in zip(params, updated_params))
    
    def get_history(self):
        return self.history