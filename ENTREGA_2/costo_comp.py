import time
from assembly import Structure
from solve import Solve

def encontrar_convergencia_deformacion_deltas(b, h, L, deltas, carga=-20000):
    """
    Itera sobre una lista de valores de delta_x = delta_y para estudiar la convergencia
    del desplazamiento vertical debido a una carga puntual o distribuida.

    Parámetros:
    - b, h, L: dimensiones de la sección y longitud de la viga (mm)
    - deltas: lista de tamaños de malla [mm]
    - carga: valor de la carga aplicada en cada nodo del extremo libre (N)

    Retorna:
    - resultados: lista de tuplas (n_elementos, uy, tiempo)
    """
    resultados = []
    uy_anterior = None

    for i, delta in enumerate(deltas):
        print(f"\n--- Iteración {i + 1} | delta = {delta:.2f} mm ---")
        inicio = time.time()

        estructura_geom = Structure(b, h, L, delta, delta)
        dof_y = [node.dofs[1] for node in estructura_geom.nodes if node.x == L]

        estructura = Solve(estructura_geom.nodes, estructura_geom.elements)

        # Aplicar carga en nodos del borde libre
        for dof in dof_y:
            estructura.apply_force(dof_index=dof, value=carga)

        estructura.solve()

        # Obtener desplazamiento vertical en el nodo de interés (último nodo)
        nodo_objetivo = estructura_geom.nodes[-1]
        uy = estructura.u_global[nodo_objetivo.dofs[1], 0]

        # Calcular número de elementos
        nodos_x = int(L / delta) + 1
        nodos_y = int(h / delta) + 1
        n_elementos = 2 * (nodos_x - 1) * (nodos_y - 1)

        tiempo = time.time() - inicio
        resultados.append((n_elementos, uy, tiempo))

        # Mostrar info
        if uy_anterior is not None:
            variacion_relativa = abs((uy - uy_anterior) / uy_anterior)
            print(f"Desplazamiento uy = {uy:.6e} mm | Variación relativa = {variacion_relativa:.2e} | Tiempo = {tiempo:.2f} s")
        else:
            print(f"Desplazamiento uy = {uy:.6e} mm | Tiempo = {tiempo:.2f} s")

        uy_anterior = uy

    return resultados
