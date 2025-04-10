import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def von_mises_stress(sigma):
    sx, sy, txy = sigma
    return np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)

def plot_full_structure(nodes, elements, u_global=None, deform_scale=0.001, alpha=0.3, dy=6):
    """
    Muestra 4 gráficos: nodos, elementos CST, estructura deformada y mapa de Von Mises.
    """
    fig, axs = plt.subplots(2, 2, figsize=(18, dy))

    # --- Subplot 1: Nodos ---
    ax = axs[0, 0]
    for node in nodes:
        ax.plot(node.x, node.y, 'bo', markersize=1.5)
    ax.set_title('Distribución de Nodos')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.axis('equal')

    # --- Subplot 2: Elementos CST ---
    ax = axs[0, 1]
    for elem in elements:
        coords = elem.get_xy_matrix()
        x = coords[:, 0]
        y = coords[:, 1]
        ax.fill(x, y, color=elem.color, alpha=alpha, edgecolor='k')
        coords_closed = np.vstack([coords, coords[0]])
        ax.plot(coords_closed[:, 0], coords_closed[:, 1], 'k-', linewidth=0.5)
    ax.set_title('Elementos CST con color')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.axis('equal')

    # --- Subplot 3: Estructura deformada ---
    ax = axs[1, 0]
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
            ax.plot(deform_coords[:, 0], deform_coords[:, 1], 'b-', linewidth=1)
        ax.set_title(f'Estructura deformada (escala x{deform_scale})')
    else:
        ax.set_title('Estructura deformada\n(no se proporcionó u_global)')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.axis('equal')

        # --- Subplot 4: Von Mises interpolado ---
    ax = axs[1, 1]
    if u_global is not None:
        centros = []
        vm_values = []
        for elem in elements:
            stress = elem.get_stress(u_global)
            vm = von_mises_stress(stress)
            centros.append(elem.get_centroid())
            vm_values.append(vm)

        centros = np.array(centros)
        vm_values = np.array(vm_values)

        # Crear malla regular para interpolar
        xi = np.linspace(centros[:, 0].min(), centros[:, 0].max(), 300)
        yi = np.linspace(centros[:, 1].min(), centros[:, 1].max(), 300)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolación de los valores de Von Mises sobre la malla
        Z = griddata(centros, vm_values, (Xi, Yi), method='cubic')

        # Dibujar mapa interpolado
        im = ax.contourf(Xi, Yi, Z, levels=100, cmap='plasma')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Von Mises [MPa]')

        ax.set_title('Mapa de tensión de Von Mises (interpolado)')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.axis('equal')
        ax.grid(True)
    else:
        ax.set_title('Von Mises (requiere u_global)')


    plt.tight_layout()
    plt.show()

def comp_cost (resultados):
    # Convertir lista de tuplas en arrays separados
    n_elems, uys, tiempos = zip(*resultados)

    n_elems = np.array(n_elems)
    uys = np.array(uys)
    tiempos = np.array(tiempos)

    # La última uy como referencia
    uy_ref = uys[-1]

    # Error relativo
    errores = np.abs((uys - uy_ref) / uy_ref)

    # Graficar con dos ejes Y
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eje izquierdo: error relativo
    color1 = 'tab:blue'
    ax1.set_xlabel('Número de elementos de la malla')
    ax1.set_ylabel('Error relativo del desplazamiento', color=color1)
    ax1.plot(n_elems, errores, 'o-', color=color1, label='Error relativo')
    ax1.tick_params(axis='y', labelcolor=color1)




    # Eje derecho: tiempo de cómputo
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Tiempo de cómputo [s]', color=color2)
    ax2.plot(n_elems, tiempos, 's--', color=color2, label='Tiempo de cómputo')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Título y leyenda combinada
    fig.suptitle('Convergencia del error y costo computacional')
    fig.tight_layout()
    plt.grid(True)
    plt.show()