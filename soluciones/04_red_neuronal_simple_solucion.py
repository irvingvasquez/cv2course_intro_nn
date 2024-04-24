# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np

def combinacion_lineal (X , W , b):
    h = np.dot(W,X) + b
    return h

# función de activación
def sigmoide(h):
    result = 1/(1+np.exp(-h))
    return result

def neurona(W, X, b, activacion):
    h = combinacion_lineal(W, X, b)
    return activacion(h)

def main():
    print("Red neuronal simple")

    entradas = np.array([1, 1])
    print("X: ", entradas)
    pesos = np.array([.2, .2])
    print("W: ", pesos)
    sesgo = 1
    print("b: ", sesgo)
    
    # inferencia o pase frontal
    activacion = sigmoide
    output = neurona(pesos, entradas, sesgo, activacion)

    print("Resultado: ", output)

if __name__ == "__main__":
    main()