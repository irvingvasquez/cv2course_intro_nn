# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

# importamos paquetes
import numpy as np

# función de activación
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivada de f
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# función h lineal
def function_h(X, W, b):
    return np.dot(W, X) + b

# Salida de la RN
def neurona(X,W,b):
    return sigmoid(function_h(X,W,b))

# Término de error
def error_term(y,W,X,b):
    error = y - neurona(X,W,b)
    delta = error * sigmoid_prime(function_h(X,W,b))
    return delta

# Incremento de los pesos
def incremento(W, X, b, eta, i, y):
    incre = eta * error_term(y,W,X,b) * X[i]
    return incre

def main():
    print("Descenso por gradiente un solo paso")

    # valores de ejemplo
    learning_rate = 1.0
    x = np.array([1,1])
    y = 1.0

    # pesos iniciales
    w = np.array([0.1,0.2])
    b = 0

    # Calcular la salida de la red
    salida = neurona(x, w, b)
    print('Salida:', salida)

    # Calcula el error residual de la red
    residual = y - salida
    print('Error residual:', residual)

    # Calcula el incremento de los pesos
    Delta_w = [incremento(w, x, b, learning_rate, 0, y), incremento(w, x, b, learning_rate, 1, y)]
    print('Incremento:', Delta_w)

    # Calcula el nuevo valor del los pesos
    n_w = w + Delta_w
    print('Nuevos pesos:', n_w)

    # Calcula el nuevo error
    n_error = y - neurona(x, n_w, b)
    print('Nuevo error residual:', n_error)

if __name__ == "__main__":
    main()