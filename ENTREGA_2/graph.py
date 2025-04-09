import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def plot_full_structure(nodes, elements, u_global=None, deform_scale=0.001, alpha=0.3, dy = 3):
    """
    Combina 3 gráficos: nodos, elementos CST y estructura deformada.
    
    Parámetros:
    - nodes: lista de objetos Node
    - elements: lista de objetos CST
    - u_global: vector de desplazamientos globales (opcional)
    - deform_scale: escala para la deformación
    - alpha: transparencia para los elementos
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, dy))

    # --- Subplot 1: Nodos ---
    ax = axs[0]
    for node in nodes:
        ax.plot(node.x, node.y, 'bo')
        #ax.text(node.x + 1, node.y + 1, str(node.id), fontsize=9, color='red')
    ax.set_title('Distribución de Nodos')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.axis('equal')

    # --- Subplot 2: Elementos CST ---
    ax = axs[1]
    for elem in elements:
        coords = elem.get_xy_matrix()
        x = coords[:, 0]
        y = coords[:, 1]
        ax.fill(x, y, color=elem.color, alpha=alpha, edgecolor='k')
        coords_closed = np.vstack([coords, coords[0]])
        ax.plot(coords_closed[:, 0], coords_closed[:, 1], 'k-', linewidth=0.5)
        centroid = elem.get_centroid()
        #ax.text(centroid[0], centroid[1], f'E{elem.element_tag}', fontsize=8, color='red')
        for node in elem.node_list:
            pass
            #ax.plot(node.x, node.y, 'bo')
            #ax.text(node.x + 0.5, node.y + 0.5, f'N{node.id}', fontsize=7, color='blue')
    ax.set_title('Elementos CST con color')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.axis('equal')

    # --- Subplot 3: Estructura deformada (si hay desplazamientos) ---
    ax = axs[2]
    if u_global is not None:
        for elem in elements:
            coords = elem.get_xy_matrix()
            deform_coords = []
            for node in elem.node_list:
                ux = u_global[node.dofs[0], 0]
                uy = u_global[node.dofs[1], 0]
                deform_coords.append([node.x + deform_scale * ux, node.y + deform_scale * uy])
            deform_coords = np.array(deform_coords)
            coords = np.vstack([coords, coords[0]])
            deform_coords = np.vstack([deform_coords, deform_coords[0]])
            ax.plot(coords[:, 0], coords[:, 1], 'gray', linewidth=0.5, linestyle='--')
            ax.plot(deform_coords[:, 0], deform_coords[:, 1], 'b-', linewidth=1.5)
        ax.set_title(f'Estructura deformada (escala x{deform_scale})')
    else:
        ax.set_title('Estructura deformada\n(no se proporcionó u_global)')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.axis('equal')

    plt.tight_layout()
    plt.show()
