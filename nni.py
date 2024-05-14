
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


def sphere_function(x, y):
    return x**2 + y**2

def sphere_gradient_x(x, y):
    return 2 * x

def sphere_gradient_y(x, y):
    return 2 * y


def ackley_function(x, y):
    return -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) - math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + 20 + math.e

def ackley_gradient_x(x, y):
    return 2 * x * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) / math.sqrt(0.5 * (x**2 + y**2)) + 2 * math.pi * math.sin(2 * math.pi * x) * math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))

def ackley_gradient_y(x, y):
    return 2 * y * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) / math.sqrt(0.5 * (x**2 + y**2)) + 2 * math.pi * math.sin(2 * math.pi * y) * math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))


class optimizationProblem2D:
    def __init__(self, function, gradient_x, gradient_y, initial_params = [0, 0], search_space = [0, 0, 0, 0], optimal_params = [0, 0]):    
        self.function = function
        self.gradient_x = gradient_x
        self.gradient_y = gradient_y
        self.initial_params = initial_params
        self.name = 'Optimization problem'
        self.optimal_params = optimal_params
        #search space
        self.min_x = search_space[0]
        self.max_x = search_space[1]
        self.min_y = search_space[2]
        self.max_y = search_space[3]

    def set_initial_params(self, initial_params):
        self.initial_params = initial_params


class BealeProblem(optimizationProblem2D):
    def __init__(self):
        super().__init__(beale_function, beale_gradient_x, beale_gradient_y, initial_params = [0, 0], \
                         search_space = [-4.5, 4.5, -4.5, 4.5], optimal_params=[3, 0.5])
        self.name = 'Función Beale'


class SphereProblem(optimizationProblem2D):
    def __init__(self):
        super().__init__(sphere_function, sphere_gradient_x, sphere_gradient_y, initial_params = [0, 0], \
                         search_space = [-2, 2, -2, 2], optimal_params=[0, 0])
        self.name = 'Función esferica'

class AckleyProblem(optimizationProblem2D):
    def __init__(self):
        super().__init__(ackley_function, ackley_gradient_x, ackley_gradient_y, initial_params = [0, 0], \
                          search_space = [-5, 5, -5, 5], optimal_params=[0, 0])
        self.name = 'Función Ackley'


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