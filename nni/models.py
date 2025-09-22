import numpy as np

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
    