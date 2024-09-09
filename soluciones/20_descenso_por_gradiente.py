
#función objetivo:
def f(x):
    return x**2

# derivada de la función objetivo:
def df(x):
    return 2*x

# descenso por gradiente:
def descenso_por_gradiente(parametro, tasa):
    parametro = parametro - tasa * df(parametro)
    return parametro

# función principal:
def main(iterations):
    print("Introducción al descenso por gradiente")

    parametro = 5
    tasa = 0.1

    print(f"Valor inicial x: {parametro}")
    print(f"Valor de f(x): {f(parametro)}")

    for i in range(iterations):
        parametro = descenso_por_gradiente(parametro, tasa)
        print(f"\nIteración: {i+1}")
        print(f"Valor de x: {parametro}")
        print(f"Valor de f(x): {f(parametro)}")

 
if __name__ == "__main__":
    iterations = int(input("Introduce la cantidad de iteraciones: "))
    if iterations <= 0:
        print("El número de iteraciones debe ser mayor a 0")
    else:
        main(iterations)
    