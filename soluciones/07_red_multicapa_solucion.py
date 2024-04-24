# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np
from nni import sigmoid, sigmoid_prime

def main():
    # Tamaño de la red
    N_input = 4
    N_hidden = 3
    N_output = 2

    #Definir matrices de pesos, inicializados de forma aleatoria
    mean = 0.0
    stdev = 0.1

    W_1 = np.random.normal(mean, scale=stdev, size=(N_input, N_hidden))
    W_2 = np.random.normal(mean, scale=stdev, size=(N_hidden, N_output))

    # Probaremos con una entrada aleatoria
    X = np.random.randn(4)

    # Ejecutar pase frontal de la red
    H_1 = np.dot(X, W_1)
    A_1 = sigmoid(H_1)

    print('Salida de la capa intermedia:')
    print(A_1)

    H_1 = np.dot(A_1, W_2)
    Y_prediccion = sigmoid(H_1)

    # Imprimir predicción
    print('Predicción de la red:')
    print(Y_prediccion)
    

if __name__ == "__main__":
    main()