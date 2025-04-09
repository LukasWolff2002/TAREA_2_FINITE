import matplotlib.pyplot as plt
import numpy as np

def plot_nodes(nodes):
    plt.figure(figsize=(8, 4))
    for node in nodes:
        plt.plot(node.x, node.y, 'bo')  # nodo como punto azul
        plt.text(node.x + 1, node.y + 1, str(node.id), fontsize=9, color='red')  # ID al lado del nodo

    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.title('Distribución de Nodos')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_sections(elements, alpha=0.5):
    plt.figure(figsize=(10, 5))

    for elem in elements:
        coords = elem.get_xy_matrix()
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Relleno del triángulo con color y opacidad
        plt.fill(x, y, color=elem.color, alpha=alpha, edgecolor='k')

        # Trazar borde del triángulo
        coords_closed = np.vstack([coords, coords[0]])  # cerrar triángulo
        plt.plot(coords_closed[:, 0], coords_closed[:, 1], 'k-', linewidth=0.5)

        # Mostrar ID del elemento en el centroide
        centroid = elem.get_centroid()
        plt.text(centroid[0], centroid[1], f'E{elem.element_tag}', fontsize=8, color='red')

        # Mostrar nodos
        for node in elem.node_list:
            plt.plot(node.x, node.y, 'bo')
            plt.text(node.x + 0.5, node.y + 0.5, f'N{node.id}', fontsize=7, color='blue')

    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Elementos CST con color")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_deformed_structure(nodes, elements, u_global, scale=0.001):
    """
    Grafica la estructura deformada sobre la malla original.
    
    Parámetros:
    - nodes: lista de objetos Node
    - elements: lista de objetos CST
    - u_global: vector de desplazamientos globales
    - scale: factor de escala para exagerar la deformación visualmente
    """
    plt.figure(figsize=(10, 5))

    for elem in elements:
        # Coordenadas originales
        coords = elem.get_xy_matrix()

        # Coordenadas deformadas
        deform_coords = []
        for node in elem.node_list:
            ux = u_global[node.dofs[0], 0]
            uy = u_global[node.dofs[1], 0]
            deform_coords.append([node.x + scale * ux, node.y + scale * uy])
        deform_coords = np.array(deform_coords)

        # Cerrar triángulo
        coords = np.vstack([coords, coords[0]])
        deform_coords = np.vstack([deform_coords, deform_coords[0]])

        # Plot original
        plt.plot(coords[:, 0], coords[:, 1], 'gray', linewidth=0.5, linestyle='--')

        # Plot deformado
        plt.plot(deform_coords[:, 0], deform_coords[:, 1], 'b-', linewidth=1.5)

    plt.title("Estructura deformada (escala x{})".format(scale))
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

