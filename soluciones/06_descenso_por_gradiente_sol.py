# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np
from data_prep import features, targets, features_test, targets_test
import matplotlib.pyplot as plt
from nni import sigmoid, sigmoid_prime
import argparse

np.random.seed(42)

def descenso_por_gradiente(ejemplos, valores_objetivo, tasa_de_aprendizaje, epocas):
    # Variables varias
    n_ejemplos, n_caracteristicas = ejemplos.shape
    Historial_error = []

    # Inicialización de pesos
    pesos = np.random.normal(scale=1 / n_caracteristicas**.5, size=n_caracteristicas)

    for e in range(epocas):
        incremento_w = np.zeros(pesos.shape)
        for x, y in zip(ejemplos.values, valores_objetivo):
            # Inferencia
            h = np.dot(x, pesos)
            salida = sigmoid(h)

            # Cálculo de error
            error = y - salida

            # Termino de error
            delta = error*sigmoid_prime(h)

            # Acumulación del gradiente
            incremento_w += delta * x

        # Actualización de pesos
        m = len(ejemplos.values)
        pesos += (tasa_de_aprendizaje/m) * incremento_w

        # Registro de error
        out = sigmoid(np.dot(ejemplos, pesos))
        error = np.mean((out - valores_objetivo) ** 2)
        Historial_error.append(error)
        if e % (epocas / 10) == 0:
            print("Época:", e, " Error: {:.3f}".format(error))
    
    return pesos, Historial_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenamiento de red neuronal con descenso por gradiente.')
    parser.add_argument('--epocas', type=int, default=200, help='Número de épocas para el entrenamiento')
    parser.add_argument('--tasa', type=float, default=0.5, help='Tasa de aprendizaje para el entrenamiento')
    args = parser.parse_args()

    # Hiperparámetros
    epocas = args.epocas
    tasa_de_aprendizaje = args.tasa

    # Entrenamiento
    pesos, Historial_error = descenso_por_gradiente(features, targets, tasa_de_aprendizaje, epocas)

    # Calcular exactitud en el conjunto de prueba
    out = sigmoid(np.dot(features_test, pesos))
    predicciones = out > 0.5
    exactitud = np.mean(predicciones == targets_test)
    print("Exactitud de la predicción: {:.3f}".format(exactitud))

