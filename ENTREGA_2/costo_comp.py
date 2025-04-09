import time
from assembly import Structure
from solve import Solve

def encontrar_convergencia_deformacion(b, h, L, delta_inicial=100, tolerancia=1.0e-10, carga=-20000, max_iter=4):
    resultados = []
    delta_x = delta_y = delta_inicial
    uy_anterior = None

    for iteracion in range(max_iter):

        if iteracion + 1 == max_iter:
            print("⚠️ Máximo número de iteraciones alcanzado.")
            break
        print(f"\n--- Iteración {iteracion + 1} | delta = {delta_x:.2f} mm ---")
        inicio = time.time()

        # Crear estructura con malla actual
        estructura_geom = Structure(b, h, L, delta_x, delta_y)
        dof_y = [node.dofs[1] for node in estructura_geom.nodes if node.x == L]

        estructura = Solve(estructura_geom.nodes, estructura_geom.elements)

        # Aplicar fuerza vertical en todos los nodos del borde libre
        for dof in dof_y:
            estructura.apply_force(dof_index=dof, value=carga)

        estructura.solve()

        # Obtener desplazamiento vertical en el último nodo
        nodo_objetivo = estructura_geom.nodes[-1]
        uy = estructura.u_global[nodo_objetivo.dofs[1], 0]

        # Calcular número de elementos
        nodos_x = int(L / delta_x) + 1
        nodos_y = int(h / delta_y) + 1
        n_elementos = 2 * (nodos_x - 1) * (nodos_y - 1)

        tiempo = time.time() - inicio
        resultados.append((n_elementos, uy, tiempo))

        if uy_anterior is not None:
            variacion_relativa = abs((uy - uy_anterior) / uy_anterior)
            print(f"Desplazamiento uy = {uy:.6e} mm | Variación relativa = {variacion_relativa:.2e} | Tiempo = {tiempo:.2f} s")

            if variacion_relativa < tolerancia:
                print("✅ Convergencia alcanzada.")
                break
        else:
            print(f"Desplazamiento uy = {uy:.6e} mm | Tiempo = {tiempo:.2f} s")

        uy_anterior = uy
        delta_x /= 2
        delta_y /= 2

        if delta_x <= 1 or delta_y <= 1:
            print("⚠️ Malla mínima alcanzada.")
            break

    return resultados
