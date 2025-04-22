import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

def plot_all_elements(elements, title, show_ids=True):
    all_x = []
    all_y = []

    # Recopilar coordenadas antes de crear la figura
    for elem in elements:
        coords = elem.get_xy_matrix()
        coords = np.vstack([coords, coords[0]])
        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])

    # Definir márgenes y límites
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    # Fijar ancho (en pulgadas), y escalar alto proporcionalmente
    fixed_width = 8  # Puedes cambiarlo
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))

    # Dibujar elementos
    for elem in elements:
        coords = elem.get_xy_matrix()
        coords = np.vstack([coords, coords[0]])
        ax.plot(coords[:, 0], coords[:, 1], 'k-', linewidth=1)

        if show_ids:
            for nodo, (x, y) in zip(elem.node_list, coords[:-1]):
                if np.all(nodo.restrain == [1, 1]):
                    pass
                    # ax.text(x, y, f'N{nodo.id}', color='blue', fontsize=6, ha='center', va='center')

    # Ajustar límites y aspecto
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Todos los elementos CST")
    ax.grid(True)

    # Guardar imagen recortando al contenido real
    plt.savefig(f"INFORME/GRAFICOS/{title}_elementos.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_applied_forces(nodes, elements, title, f_vector, scale=1e-2):
    all_x = []
    all_y = []

    # Recolectar coordenadas de todos los elementos
    for elem in elements:
        coords = elem.get_xy_matrix()
        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])

    # También considerar nodos, por si hay fuerzas fuera del dominio de elementos
    for node in nodes:
        all_x.append(node.x)
        all_y.append(node.y)

    # Calcular límites con márgenes
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    # Escalar altura según ancho fijo
    fixed_width = 8  # pulgadas
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))
    ax.set_title("Fuerzas aplicadas sobre los nodos")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Dibujar elementos
    for elem in elements:
        coords = elem.get_xy_matrix()
        coords = np.vstack([coords, coords[0]])
        ax.plot(coords[:, 0], coords[:, 1], color='lightgray', linewidth=1, zorder=1)

    # Dibujar fuerzas
    for node in nodes:
        dof_x, dof_y = node.dofs
        fx = f_vector[dof_x][0] if dof_x < len(f_vector) else 0.0
        fy = f_vector[dof_y][0] if dof_y < len(f_vector) else 0.0

        if not np.isclose(fx, 0.0) or not np.isclose(fy, 0.0):
            ax.quiver(node.x, node.y, fx, fy,
                      angles='xy', scale_units='xy',
                      scale=1 / scale, color='red', width=0.005,
                      zorder=5)

    # Guardar sin márgenes blancos
    plt.savefig(f"INFORME/GRAFICOS/{title}_fuerzas.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_deformed_structure(elements, title, scale=1.0, show_ids=False):
    all_x = []
    all_y = []

    # Primero recopilamos todas las coordenadas deformadas y originales
    for elem in elements:
        coords = elem.get_xy_matrix()
        u = np.zeros_like(coords)

        for i, nodo in enumerate(elem.node_list):
            ux = nodo.structure.u_global[nodo.dofs[0], 0]
            uy = nodo.structure.u_global[nodo.dofs[1], 0]
            u[i] = [ux, uy]

        def_coords = coords + scale * u

        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])
        all_x.extend(def_coords[:, 0])
        all_y.extend(def_coords[:, 1])

    # Margen y proporción para figura
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    fixed_width = 8  # pulgadas
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))

    # Ahora sí graficamos
    for elem in elements:
        coords = elem.get_xy_matrix()
        u = np.zeros_like(coords)

        for i, nodo in enumerate(elem.node_list):
            ux = nodo.structure.u_global[nodo.dofs[0], 0]
            uy = nodo.structure.u_global[nodo.dofs[1], 0]
            u[i] = [ux, uy]

        coords_closed = np.vstack([coords, coords[0]])
        def_coords = coords + scale * u
        def_coords_closed = np.vstack([def_coords, def_coords[0]])

        ax.plot(coords_closed[:, 0], coords_closed[:, 1], 'gray', linewidth=0.5)
        ax.plot(def_coords_closed[:, 0], def_coords_closed[:, 1], 'b', linewidth=1.0)

        if show_ids:
            for i, nodo in enumerate(elem.node_list):
                x, y = def_coords[i]
                ax.text(x, y, str(nodo.id), fontsize=6, color='red')

    # Ajustar límites y aspecto
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title(f"Estructura deformada (amplificación ×{scale})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Guardar sin espacios en blanco
    plt.savefig(f"INFORME/GRAFICOS/{title}_deformada.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_deformed_with_reactions(title, elements, reactions, scale=1.0, reaction_scale=1e-3, show_ids=False):
    all_x = []
    all_y = []

    for elem in elements:
        coords = elem.get_xy_matrix()
        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    fixed_width = 16
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio / 2

    fig, axs = plt.subplots(1, 2, figsize=(fixed_width, height))
    ax1, ax2 = axs

    # === GRÁFICO 1: Reacciones solamente ===
    for elem in elements:
        coords = elem.get_xy_matrix()
        coords_closed = np.vstack([coords, coords[0]])
        ax1.plot(coords_closed[:, 0], coords_closed[:, 1], 'lightgray', linewidth=0.5)

        for i, nodo in enumerate(elem.node_list):
            rx = reactions[nodo.dofs[0]][0] if nodo.restrain[0] == 1 else 0.0
            ry = reactions[nodo.dofs[1]][0] if nodo.restrain[1] == 1 else 0.0
            if rx != 0.0 or ry != 0.0:
                ax1.quiver(nodo.x, nodo.y, rx, ry, angles='xy', scale_units='xy',
                           scale=1/reaction_scale, color='red', width=0.005)
            if show_ids:
                ax1.text(nodo.x, nodo.y, f'N{nodo.id}', fontsize=6)

    ax1.set_title(f"Reacciones nodales", fontsize=14)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True)
    ax1.set_xlim(x_min - x_margin, x_max + x_margin)
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)
    ax1.set_aspect('equal', adjustable='box')

    # === GRÁFICO 2: Mapa de calor de reacciones ===
    nodal_values = {}
    for elem in elements:
        for nodo in elem.node_list:
            rx = reactions[nodo.dofs[0]][0] if nodo.restrain[0] == 1 else 0.0
            ry = reactions[nodo.dofs[1]][0] if nodo.restrain[1] == 1 else 0.0
            mag = np.sqrt(rx**2 + ry**2)
            nodal_values[nodo.id] = mag

    all_nodes = {n.id: n for e in elements for n in e.node_list}
    node_list = list(all_nodes.values())
    node_id_map = {node.id: i for i, node in enumerate(node_list)}
    points = np.array([[node.x, node.y] for node in node_list])
    triangles = []
    values = []

    for elem in elements:
        ids = [n.id for n in elem.node_list]
        val = np.mean([nodal_values.get(nid, 0.0) for nid in ids])
        if val > 0:
            tri = [node_id_map[nid] for nid in ids]
            triangles.append(tri)
            values.append(val)

    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    for elem in elements:
        coords = elem.get_xy_matrix()
        coords = np.vstack([coords, coords[0]])
        ax2.plot(coords[:, 0], coords[:, 1], color='lightgray', linewidth=0.5)

    if values:
        tpc = ax2.tripcolor(triang, facecolors=values, edgecolors='k', cmap='plasma', zorder=2)
        cbar = fig.colorbar(tpc, ax=ax2)
        cbar.set_label("Reaccion Force [N]", fontsize=14)

    ax2.set_title("Mapa de calor de reacciones por elemento", fontsize=14)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True)
    ax2.set_xlim(x_min - x_margin, x_max + x_margin)
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    ax2.set_aspect('equal', adjustable='box')

    fig.savefig(f"INFORME/GRAFICOS/{title}_deformada_reacciones.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_von_mises_field(nodes, elements, vm_nodal_dict, title, cmap='plasma'):
    node_id_to_index = {}
    xs, ys, vms = [], [], []

    for i, node in enumerate(nodes):
        node_id_to_index[node.id] = i
        xs.append(node.x)
        ys.append(node.y)
        vms.append(vm_nodal_dict.get(node.id, 0.0))

    triangles = []
    for elem in elements:
        triangle = [node_id_to_index[n.id] for n in elem.node_list]
        triangles.append(triangle)

    triang = mtri.Triangulation(xs, ys, triangles)

    # Calcular límites para adaptar proporción del gráfico
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    fixed_width = 8  # pulgadas
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))

    tcf = ax.tricontourf(triang, vms, levels=20, cmap=cmap)
    ax.triplot(triang, color='gray', linewidth=0.5)
    cbar = fig.colorbar(tcf, ax=ax)
    cbar.set_label("Tensión de Von Mises (Pa)")

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Campo de Von Mises sobre los elementos")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Guardar imagen recortada y centrada
    fig.savefig(f"INFORME/GRAFICOS/{title}_von_mises.png", dpi=300, bbox_inches='tight')
    plt.close()
