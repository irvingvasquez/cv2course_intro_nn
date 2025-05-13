# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

import numpy as np

def saludar():
    print("Hola desde modulo1")


def correlacionPixel(H, I, i, j):
    #Operacion de correlacion para el pixel i , j
    # Determinar el tamano del kernel
    m, n = I.shape
    doskmas1, _ = H.shape
    k = np.floor (( doskmas1 - 1 ) / 2 ).astype ( int )
    # Inicializar sumatoria
    sumatoria = 0
    # Recorrido del Kernel y la imagen
    for u in range (-k, k + 1 ):
        for v in range (-k , k + 1):
            sumatoria +=H [ u +k , v + k ] * I [ i +u , j + v ]
    
    return sumatoria . astype ( int )


def correlacionCruzadaValida(H, I):
    doskmas1, _ = H.shape
    # Se determina el valor de k y se guarda como tipo entero
    k = np . floor (( doskmas1 - 1 ) / 2 ) . astype ( int )
    m, n = I.shape
    G = np.zeros ((m ,n))

    # Correlación para cada elemento de I, nota que en la matriz G solo se llenan los elementos donde cabe el kernel
    for i in range(k, m-k):
        for j in range(k, n-k):
            G[i,j] = correlacionPixel(H,I,i,j)
    return G