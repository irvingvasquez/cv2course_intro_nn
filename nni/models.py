# Author: Juan Irving Vasquez (jvasquezg@ipn.mx)
# Date: 2024-06-01
# License: Attribution 4.0 International

import numpy as np
from abc import ABC, abstractmethod

def neuronaMyP(E,I,u):
    for inhibitoria in I:
        if inhibitoria == 1:
            return 0
    
    integracion = 0
    for exitatoria in E:
        integracion = integracion + exitatoria
    
    if integracion >= u:
        return 1
    else:
        return 0
    

# clase base Neurona
class Neurona:
    def __init__(self, W, b, activacion):
        self.W = W
        self.b = b
        self.activacion = activacion

    def combinacion_lineal(self, X):
        h = np.dot(X, self.W) + self.b
        return h

    def forward(self, X):
        h = self.combinacion_lineal(X)
        return self.activacion(h)
    
# clase base abstracta para incluir los modelos de redes
class Modelo(ABC):
    def __init__(self):
        self.n_parametros = 0

    @abstractmethod
    def forward(self, X):
        pass

class Lineal(Modelo):
    """
    Capa lineal (totalmente conectada) de una red neuronal.

    Esta clase implementa una transformación lineal seguida de una función de activación.
    Hereda de la clase Modelo.

    Attributes:
        W (np.ndarray): Matriz de pesos de forma (n_inputs, n_outputs).
        b (np.ndarray): Vector de sesgos de forma (n_outputs,).
        activacion (callable): Función de activación a aplicar a la salida lineal.
        n_parametros (int): Número total de parámetros (pesos + sesgos).

    Methods:
        __init__(n_inputs, n_outputs, activacion):
            Inicializa la capa lineal con pesos y sesgos aleatorios.
            
            Args:
                n_inputs (int): Número de características de entrada.
                n_outputs (int): Número de unidades de salida.
                activacion (callable): Función de activación a aplicar.
        
        forward(X):
            Calcula la salida de la capa para un lote de entradas.
            
            Args:
                X (np.ndarray): Matriz de entrada de forma (batch_size, n_inputs).
            
            Returns:
                np.ndarray: Salida después de aplicar la transformación lineal
                           y la función de activación.
    """
    def __init__(self, W, b, activacion):
        super().__init__()
        self.W = W
        self.b = b
        self.activacion = activacion
        self.n_parametros = W.size + b.size

    def __init__(self, n_inputs, n_outputs, activacion):
        super().__init__()
        self.W = np.random.rand(n_inputs, n_outputs)
        self.b = np.random.rand(n_outputs)
        self.activacion = activacion
        self.n_parametros = self.W.size + self.b.size

    def forward(self, X):
        h = np.dot(X, self.W) + self.b
        return self.activacion(h)
    


class RedMulticapa(Modelo):
    def __init__(self, capas):
        super().__init__()
        self.capas = capas
        self.n_parametros = sum(capa.n_parametros for capa in capas)

    def forward(self, X):
        for capa in self.capas:
            X = capa.forward(X)
        return X

