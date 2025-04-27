import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import os
import re

def clean_filename(text):
    """Convierte un string en nombre de archivo válido eliminando LaTeX y símbolos especiales."""
    return re.sub(r'[^\w\-_. ]', '', text.replace('$', '').replace('\\', '').replace('{', '').replace('}', ''))

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
    ax.set_title("All CST elements")

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
    ax.set_title("Forces applied on nodes")

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

    ax.set_title(f"Deformed structure (amplification ×{scale})")

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

    ax1.set_title(f"Nodal reactions", fontsize=14)

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
        cbar.set_label("Reaction force [N]", fontsize=14)


    ax2.set_title("Heatmap of reactions per element", fontsize=14)

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
    #ax.triplot(triang, color='gray', linewidth=0.5)
    cbar = fig.colorbar(tcf, ax=ax)
    cbar.set_label("Von Mises Stress (Pa)")

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Von Mises stress field over elements")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Guardar imagen recortada y centrada
    fig.savefig(f"INFORME/GRAFICOS/{title}_von_mises.png", dpi=300, bbox_inches='tight')
    plt.close()

from matplotlib.colors import BoundaryNorm

def plot_von_mises_per_element(nodes, elements, vm_nodal_dict, title, cmap='plasma'):
    node_id_to_index = {}
    xs, ys = [], []

    for i, node in enumerate(nodes):
        node_id_to_index[node.id] = i
        xs.append(node.x)
        ys.append(node.y)

    triangles = []
    element_colors = []

    for elem in elements:
        triangle = [node_id_to_index[n.id] for n in elem.node_list]
        triangles.append(triangle)
        
        # Usamos promedio de los nodos para el valor del elemento
        vms_elem = np.mean([vm_nodal_dict.get(n.id, 0.0) for n in elem.node_list])
        element_colors.append(vms_elem)

    triang = mtri.Triangulation(xs, ys, triangles)

    # --- Escalado proporcional ---
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    fixed_width = 8
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))

    # --- Discretizar en 11 colores ---
    n_colors = 11
    vmin = min(element_colors)
    vmax = max(element_colors)
    levels = np.linspace(vmin, vmax, n_colors)
    norm = BoundaryNorm(boundaries=levels, ncolors=n_colors-1)

    # 🎯 Graficar
    tpc = ax.tripcolor(triang, facecolors=element_colors, edgecolors='k', cmap=cmap, norm=norm)

    cbar = fig.colorbar(tpc, ax=ax, ticks=levels)
    cbar.set_label("Von Mises Stress (Pa)")
    cbar.ax.set_yticklabels([f"{lvl:.2e}" for lvl in levels])  # Formato de etiquetas

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Von Mises stress per element (11 colors)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    fig.savefig(f"INFORME/GRAFICOS/{title}_von_mises_per_element_11_colors.png", dpi=300, bbox_inches='tight')
    plt.close()


def compute_nodal_principal_fields(nodes, elements, u_global):
    """
    Calcula los esfuerzos y deformaciones principales σ1, σ2, ε1, ε2 por nodo.
    Devuelve 4 diccionarios: sigma1, sigma2, eps1, eps2
    """
    sigma1_map = {node.id: [] for node in nodes}
    sigma2_map = {node.id: [] for node in nodes}
    eps1_map = {node.id: [] for node in nodes}
    eps2_map = {node.id: [] for node in nodes}

    for elem in elements:
        σ = elem.get_stress(u_global)
        ε = elem.get_strain(u_global)

        σx, σy, τxy = σ
        εx, εy, γxy = ε

        # Esfuerzos principales
        σ_avg = 0.5 * (σx + σy)
        Rσ = np.sqrt(((σx - σy) / 2)**2 + τxy**2)
        σ1, σ2 = σ_avg + Rσ, σ_avg - Rσ

        # Deformaciones principales
        ε_avg = 0.5 * (εx + εy)
        Rε = np.sqrt(((εx - εy) / 2)**2 + (γxy / 2)**2)
        ε1, ε2 = ε_avg + Rε, ε_avg - Rε

        for node in elem.node_list:
            sigma1_map[node.id].append(σ1)
            sigma2_map[node.id].append(σ2)
            eps1_map[node.id].append(ε1)
            eps2_map[node.id].append(ε2)

    # Promediar por nodo
    sigma1 = {nid: np.mean(vals) for nid, vals in sigma1_map.items()}
    sigma2 = {nid: np.mean(vals) for nid, vals in sigma2_map.items()}
    eps1 = {nid: np.mean(vals) for nid, vals in eps1_map.items()}
    eps2 = {nid: np.mean(vals) for nid, vals in eps2_map.items()}

    return sigma1, sigma2, eps1, eps2

def plot_all_scalar_fields_separately(nodes, elements, nodal_fields, title_prefix):
    """
    Genera 6 gráficos individuales para: σxx, σyy, τxy, εxx, εyy, γxy
    """
    σxx, σyy, τxy, εxx, εyy, γxy = {}, {}, {}, {}, {}, {}

    for node_id, (sigma, epsilon) in nodal_fields.items():
        σxx[node_id] = sigma[0]
        σyy[node_id] = sigma[1]
        τxy[node_id] = sigma[2]
        εxx[node_id] = epsilon[0]
        εyy[node_id] = epsilon[1]
        γxy[node_id] = epsilon[2]

    plot_scalar_field(nodes, elements, σxx, r"$\sigma_{xx}$ (Pa)", f"{title_prefix} - sigma_xx")
    plot_scalar_field(nodes, elements, σyy, r"$\sigma_{yy}$ (Pa)", f"{title_prefix} - sigma_yy")
    plot_scalar_field(nodes, elements, τxy, r"$\tau_{xy}$ (Pa)", f"{title_prefix} - tau_xy")
    plot_scalar_field(nodes, elements, εxx, r"$\varepsilon_{xx}$", f"{title_prefix} - epsilon_xx")
    plot_scalar_field(nodes, elements, εyy, r"$\varepsilon_{yy}$", f"{title_prefix} - epsilon_yy")
    plot_scalar_field(nodes, elements, γxy, r"$\gamma_{xy}$", f"{title_prefix} - gamma_xy")

    plot_scalar_field_per_element(nodes, elements, σxx, r"$\sigma_{xx}$ (Pa)", f"{title_prefix} - sigma_xx_per_element")
    plot_scalar_field_per_element(nodes, elements, σyy, r"$\sigma_{yy}$ (Pa)", f"{title_prefix} - sigma_yy_per_element")
    plot_scalar_field_per_element(nodes, elements, τxy, r"$\tau_{xy}$ (Pa)", f"{title_prefix} - tau_xy_per_element")
    plot_scalar_field_per_element(nodes, elements, εxx, r"$\varepsilon_{xx}$", f"{title_prefix} - epsilon_xx_per_element")
    plot_scalar_field_per_element(nodes, elements, εyy, r"$\varepsilon_{yy}$", f"{title_prefix} - epsilon_yy_per_element")
    plot_scalar_field_per_element(nodes, elements, γxy, r"$\gamma_{xy}$", f"{title_prefix} - gamma_xy_per_element")

def plot_scalar_field(nodes, elements, nodal_values, field_title, filename_prefix, cmap='plasma'):
    """
    Grafica un campo escalar interpolado sobre la malla de elementos.
    """
    node_id_to_index = {node.id: i for i, node in enumerate(nodes)}
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]

    # Triángulos de elementos
    triangles = [
        [node_id_to_index[n.id] for n in elem.node_list]
        for elem in elements
    ]

    # Valores del campo escalar por nodo
    values = np.zeros(len(nodes))
    for node in nodes:
        values[node_id_to_index[node.id]] = nodal_values[node.id]

    # Márgenes para recorte automático
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    fixed_width = 8  # en pulgadas
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))
    triang = mtri.Triangulation(xs, ys, triangles)

    tcf = ax.tricontourf(triang, values, levels=20, cmap=cmap)
    cbar = fig.colorbar(tcf, ax=ax)
    cbar.set_label(field_title)

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(field_title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Guardar imagen recortada
    os.makedirs("INFORME/GRAFICOS", exist_ok=True)
    filename = f"INFORME/GRAFICOS/{clean_filename(filename_prefix)}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

def plot_scalar_field_per_element(nodes, elements, nodal_values, field_title, filename_prefix, cmap='plasma'):
    node_id_to_index = {node.id: i for i, node in enumerate(nodes)}
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]

    triangles = []
    element_values = []

    for elem in elements:
        triangle = [node_id_to_index[n.id] for n in elem.node_list]
        triangles.append(triangle)

        # Valor promedio del elemento
        value = np.mean([nodal_values[n.id] for n in elem.node_list])

        # ⚡ Si el valor es 0 o negativo, lo reemplazamos por un pequeño valor positivo
        if value <= 0:
            value = 1e-6

        element_values.append(value)

    triang = mtri.Triangulation(xs, ys, triangles)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    fixed_width = 8
    aspect_ratio = ((y_max - y_min) + 2 * y_margin) / ((x_max - x_min) + 2 * x_margin)
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))

    vmin = min(element_values)
    vmax = max(element_values)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    tpc = ax.tripcolor(
        triang,
        facecolors=element_values,
        cmap=cmap,
        norm=norm,
        edgecolors='k',
        shading='flat'
    )

    cbar = fig.colorbar(tpc, ax=ax)
    cbar.set_label(field_title)

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(field_title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    os.makedirs("INFORME/GRAFICOS", exist_ok=True)
    filename = f"INFORME/GRAFICOS/{clean_filename(filename_prefix)}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_principal_fields(nodes, elements, u_global, title_prefix):
        """
        Genera 4 gráficos: σ1, σ2, ε1, ε2.
        """
        sigma1, sigma2, eps1, eps2 = compute_nodal_principal_fields(nodes, elements, u_global)

        plot_scalar_field(nodes, elements, sigma1, r"$\sigma_1$ (Pa)", f"{title_prefix} - sigma_1")
        plot_scalar_field(nodes, elements, sigma2, r"$\sigma_2$ (Pa)", f"{title_prefix} - sigma_2")
        plot_scalar_field(nodes, elements, eps1, r"$\varepsilon_1$", f"{title_prefix} - epsilon_1")
        plot_scalar_field(nodes, elements, eps2, r"$\varepsilon_2$", f"{title_prefix} - epsilon_2")

        plot_scalar_field_per_element(nodes, elements, sigma1, r"$\sigma_1$ (Pa)", f"{title_prefix} - sigma_1_per_element")
        plot_scalar_field_per_element(nodes, elements, sigma2, r"$\sigma_2$ (Pa)", f"{title_prefix} - sigma_2_per_element")
        plot_scalar_field_per_element(nodes, elements, eps1, r"$\varepsilon_1$", f"{title_prefix} - epsilon_1_per_element")
        plot_scalar_field_per_element(nodes, elements, eps2, r"$\varepsilon_2$", f"{title_prefix} - epsilon_2_per_element")

import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import numpy as np

def plot_elements_by_thickness(elements, title="espesores", cmap='viridis'):
    """
    Dibuja y guarda una figura coloreando los elementos según el espesor de su sección,
    ajustando automáticamente el alto para mantener proporciones.
    """
    node_ids = {}
    xs, ys = [], []
    triangles = []
    thicknesses = []

    counter = 0
    for elem in elements:
        triangle = []
        for node in elem.node_list:
            if node.id not in node_ids:
                node_ids[node.id] = counter
                xs.append(node.x)
                ys.append(node.y)
                counter += 1
            triangle.append(node_ids[node.id])
        triangles.append(triangle)
        thicknesses.append(elem.section.thickness)

    triang = mtri.Triangulation(xs, ys, triangles)

    # Cálculo de límites y dimensiones
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    fixed_width = 8
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))
    tpc = ax.tripcolor(triang, facecolors=thicknesses, edgecolors='k', cmap=cmap)
    cbar = plt.colorbar(tpc, ax=ax, label="Thickness (mm)")
    ax.set_title("Thickness Distribution per Element")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"INFORME/GRAFICOS/{title}_espesores.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico guardado como INFORME/GRAFICOS/{title}_espesores.png")
