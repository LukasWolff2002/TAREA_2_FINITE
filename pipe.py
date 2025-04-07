import numpy as np
import matplotlib.pyplot as plt

def cst(xy, properties, ue=None):
    # Extraer las propiedades del material
    E = properties["E"]  # Módulo de elasticidad
    rho = properties["rho"]  # Densidad del material
    
    # Coordenadas de los nodos del triángulo
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    # Cálculo del área del triángulo (A)
    A = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    # Funciones de forma (N) para el triángulo CST
    N1 = lambda xi, eta: 1 - xi - eta
    N2 = lambda xi, eta: xi
    N3 = lambda xi, eta: eta

    # Matriz de la deformación en función de las coordenadas naturales
    B = np.array([[-1, 0, 1, 0, 0, 0],
                  [0, -1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1]])

    # Cálculo de la matriz de rigidez (ke) y el vector de fuerzas (fe)
    if ue is None:
        # Matriz de rigidez local para el elemento triangular CST
        ke = np.zeros((6, 6))
        fe = np.zeros(6)

        # Integración sobre el área del triángulo (en este caso es exacta)
        # Ya que el triángulo tiene una forma de interpolación lineal y el área es constante
        for i in range(3):
            for j in range(3):
                ke += E * A * np.dot(B.T, B)  # Matriz de rigidez

        # Vector de fuerzas internas (este es un ejemplo de integración simple)
        fe[0] = rho * A  # Fuerzas en los nodos debido a la densidad y el área

        return ke, fe

    else:
        # Si el desplazamiento ue está presente, calcular las deformaciones y esfuerzos
        epsilon_e = np.zeros(3)
        sigma_e = np.zeros(3)

        # Calculamos las deformaciones a partir del desplazamiento nodal (ue)
        epsilon_e[0] = (ue[1] - ue[0]) / A  # Deformación en la dirección x
        epsilon_e[1] = (ue[3] - ue[2]) / A  # Deformación en la dirección y
        epsilon_e[2] = (ue[4] - ue[5]) / A  # Deformación cortante

        # Calculamos los esfuerzos a partir de las deformaciones
        sigma_e[0] = E * epsilon_e[0]  # Esfuerzo en la dirección x
        sigma_e[1] = E * epsilon_e[1]  # Esfuerzo en la dirección y
        sigma_e[2] = E * epsilon_e[2]  # Esfuerzo cortante

        return epsilon_e, sigma_e

# Ejemplo de uso

# Coordenadas de los nodos (x, y)
xy = np.array([[0, 0], [4, 0], [0, 6]])

# Propiedades del material
properties = {"E": 210000, "rho": 7800}  # Módulo de elasticidad y densidad

# Desplazamientos nodales (ue) - en caso de que queramos calcular deformaciones y esfuerzos
ue = np.array([0, 0, 0.005, 0.003, 0, 0.002])

# Llamada a la función
ke, fe = cst(xy, properties)  # Si no proporcionamos ue, se calcula la rigidez y fuerzas
epsilon_e, sigma_e = cst(xy, properties, ue)  # Si proporcionamos ue, se calculan las deformaciones y esfuerzos

# Imprimir los resultados

print("Matriz de Rigidez (ke):")
print(ke)

print("\nVector de Fuerzas (fe):")
print(fe)

print("\nDeformaciones (epsilon_e):")
print(epsilon_e)

print("\nEsfuerzos (sigma_e):")
print(sigma_e)

# Graficar el triángulo
x_coords = xy[:, 0]  # Extraer las coordenadas X de los nodos
y_coords = xy[:, 1]  # Extraer las coordenadas Y de los nodos

# Cerrar el triángulo agregando el primer nodo al final
x_coords = np.append(x_coords, x_coords[0])
y_coords = np.append(y_coords, y_coords[0])

# Crear el gráfico
plt.figure(figsize=(6, 6))
plt.plot(x_coords, y_coords, 'bo-', label="Triángulo CST")
plt.fill(x_coords, y_coords, 'cyan', alpha=0.3)

# Etiquetas y título
plt.title("Triángulo CST")
plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.grid(True)
plt.legend()

# Mostrar el gráfico
plt.show()
