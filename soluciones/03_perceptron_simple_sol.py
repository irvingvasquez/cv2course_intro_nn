# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np

def combinacion_lineal (X , W , b):
    suma = 0
    for w , x in zip (W , X ):
        suma = suma + w * x
    suma = suma + b
    return suma

def escalon ( h ):
    if h >= 0 :
        return 1
    else :
        return 0

def perceptron(W, X, b, activacion):
    h = combinacion_lineal(W, X, b)
    return activacion(h)

inputs = np.array([0.7 , -0.3 ])
weights = np.array([0.1 , 0.8 ])
bias = 0.5

# Pase frontal
activacion = escalon
output = perceptron(weights, inputs, bias, activacion)
print ( ' Output : ')
print ( output )