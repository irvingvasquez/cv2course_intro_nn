
# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np
import math
import pandas as pd
import nni.functions

def sigmoid(h):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-h))

def sigmoid_prime(h):
    return sigmoid(h) * (1-sigmoid(h))

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
vectorized_ackley_function = np.vectorize(ackley_function)

def ackley_gradient_x(x, y):
    return 2 * x * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) / math.sqrt(0.5 * (x**2 + y**2)) + 2 * math.pi * math.sin(2 * math.pi * x) * math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
vectorized_ackley_gradient_x = np.vectorize(ackley_gradient_x)

def ackley_gradient_y(x, y):
    return 2 * y * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) / math.sqrt(0.5 * (x**2 + y**2)) + 2 * math.pi * math.sin(2 * math.pi * y) * math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
vectorized_ackley_gradient_y = np.vectorize(ackley_gradient_y)


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

class ReducedBealeProblem(optimizationProblem2D):
    def __init__(self):
        super().__init__(beale_function, beale_gradient_x, beale_gradient_y, initial_params = [0, 0], \
                         search_space = [0.0, 4.5, -2.0, 2.0], optimal_params=[3, 0.5])
        self.name = 'Función Beale reducida'


class SphereProblem(optimizationProblem2D):
    def __init__(self):
        super().__init__(sphere_function, sphere_gradient_x, sphere_gradient_y, initial_params = [0, 0], \
                         search_space = [-2, 2, -2, 2], optimal_params=[0, 0])
        self.name = 'Función esferica'

class AckleyProblem(optimizationProblem2D):
    def __init__(self):
        super().__init__(vectorized_ackley_function, vectorized_ackley_gradient_x, vectorized_ackley_gradient_y, initial_params = [0, 0], \
                          search_space = [-5, 5, -5, 5], optimal_params=[0, 0])
        self.name = 'Función Ackley'


class Optimizer:
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance
        self.history = []
        self.name = 'Clase Base Optimizer'

class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        super().__init__(tolerance=tolerance)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.name = 'Descenso por gradiente'

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
    
def correlacionPixel (H , I , i , j ) :
    # Operacion de correlacion para el pixel i , j
    # Determinar el tamano del kernel
    m , n = I . shape
    doskmas1 , _ = H . shape
    k = np . floor (( doskmas1 - 1 ) / 2 ) . astype ( int )
    # Inicializar sumatoria
    sumatoria = 0
    # Recorrido del Kernel y la imagen
    for u in range ( -k , k + 1 ) :
        for v in range ( -k , k + 1 ) :
            sumatoria += H [ u +k , v + k ] * I [ i +u , j + v ]
    return sumatoria . astype ( int )

def correlacionCruzada(H, I):
	doskmas1, _ = H.shape
	k = np.floor((doskmas1 - 1)/2).astype(int)
	m, n = I.shape
	G = np.zeros((m,n))
	
	# Realiza la correlacion para cada elemento de I
	for i in range(k, m-k):
		for j in range(k, n-k):
			G[i,j] = correlacionPixel(H,I,i,j)
	return G

def ReLU(x):
    return np.maximum(0, x)

def datosSinteticos():
    gen = np.random.RandomState(1)
    mean1, cov1 = [0, 0], [[1, 0], [0, 20]]
    mean2, cov2 = [0, 20], [[1, 1], [1, 20]]
    n_samples = 400
    X, y = pd.DataFrame(np.vstack([np.random.multivariate_normal(mean1, cov1, size=int(n_samples/2)),
                               np.random.multivariate_normal(mean2, cov2, size=int(n_samples/2))]),
                    columns=['x1', 'x2']), pd.Series([0]*int(n_samples/2)+[1]*int(n_samples/2), name='target')
    return X, y
