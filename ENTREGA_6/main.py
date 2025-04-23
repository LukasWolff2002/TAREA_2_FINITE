import os
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio

from nodes import Node
from section import Section
from cst import CST
from solve import Solve

from graph import plot_all_elements, plot_applied_forces, plot_deformed_structure, plot_deformed_with_reactions, plot_von_mises_field

# ==============================
# Función para generar la malla
# ==============================
def generate_mesh(input_file, output_file, lc=5):
    gmsh.initialize()
    gmsh.open(input_file)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(output_file)
    gmsh.finalize()

# ================================================
# Función para crear nodos agrupados por secciones
# ================================================
def make_nodes_groups(output_file, title):
    mesh = meshio.read(output_file)
    tag_to_name = {v[0]: k for k, v in mesh.field_data.items()}
    grupos = {}

    # Elementos tipo "triangle"
    for cell_block, phys_tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cell_block.type != "triangle":
            continue
        for tri, tag in zip(cell_block.data, phys_tags):
            nombre = tag_to_name.get(tag, f"{tag}")
            if nombre not in grupos:
                grupos[nombre] = []
            for node_id in tri:
                x, y = mesh.points[node_id][:2]
                grupos[nombre].append(Node(node_id + 1, x, y))

    # Elementos tipo "line" (por ejemplo, para restricciones o cargas)
    for cell_block, phys_tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cell_block.type != "line":
            continue
        for line, tag in zip(cell_block.data, phys_tags):
            nombre = tag_to_name.get(tag, f"{tag}")
            if nombre not in grupos:
                grupos[nombre] = []
            for node_id in line:
                x, y = mesh.points[node_id][:2]
                restrain = [0, 0]
                if nombre in ["Restr Sup", "Restr Inf"]:
                    restrain = [1, 1]
                grupos[nombre].append(Node(node_id + 1, x, y, restrain=restrain))

    # Eliminar duplicados dentro de cada grupo
    for nombre in grupos:
        nodos_unicos = {}
        for nodo in grupos[nombre]:
            nodos_unicos[nodo.id] = nodo
        grupos[nombre] = list(nodos_unicos.values())

    # Visualizar nodos por grupo
    Node.plot_nodes_por_grupo(grupos, title, show_ids=False)

    return grupos, mesh

# ========================================
# Crear secciones del modelo y asociarlas
# ========================================
def make_sections(grupos):
    thickness = {"1": 1.6, "2": 3.0, "3": 5.0, "4": 5.0}
    sections = {}

    # Propiedades del material (ortotrópico PLA impreso)
    for group in thickness:
        sections[group] = Section(thickness[group], Ex=3000, Ey=1200, nuxy=0.35, Gxy=500)

    # Diccionario global de nodos para búsqueda por ID
    nodes_dict = {}
    for group in grupos:
        for node in grupos[group]:
            nodes_dict[node.id] = node

    return sections, nodes_dict

def make_cst_elements (mesh, sections, nodes_dict):

    triangles = mesh.cells_dict['triangle']  # nodos por cada triángulo
    tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    elements = []
    nodes = set()  # Usamos un set para evitar nodos repetidos

    for i in range(len(tags)):
        section = sections[str(tags[i])]

        node_ids = triangles[i]
        nodo_a = nodes_dict[node_ids[0]+1]
        nodo_b = nodes_dict[node_ids[1]+1]
        nodo_c = nodes_dict[node_ids[2]+1]

        for nodo in [nodo_a, nodo_b, nodo_c]:
            nodes.add(nodo)  # Se agregan al set automáticamente sin duplicados

        elem = CST(i+1, [nodo_a, nodo_b, nodo_c], section)
        elements.append(elem)

    nodes = list(nodes)

    return elements, nodes

def apply_distributed_force(grupo_nodos, fuerza_total_y, estructura):
    """
    Aplica una fuerza distribuida vertical (por ejemplo, peso) sobre una línea formada por nodos no alineados.
    La fuerza se reparte proporcionalmente a la longitud de los tramos y se descompone en x e y.
    """

    # Paso 1: ordena nodos si es necesario (aquí asumimos que ya están ordenados)
    nodos = grupo_nodos
    n = len(nodos)
    if n < 2:
        print("Se requieren al menos dos nodos para aplicar fuerza distribuida.")
        return

    # Paso 2: calcular longitud total de la línea
    longitudes = []
    total_length = 0
    for i in range(n - 1):
        dx = nodos[i+1].x - nodos[i].x
        dy = nodos[i+1].y - nodos[i].y
        L = np.sqrt(dx**2 + dy**2)
        longitudes.append(L)
        total_length += L

    # Paso 3: calcular carga distribuida por unidad de longitud
    q_total = fuerza_total_y  # Fuerza total a repartir
    q_lineal = q_total / total_length  # N/m

    # Paso 4: aplicar cargas parciales a cada nodo (2 nodos por segmento)
    nodal_forces = {node.id: np.array([0.0, 0.0]) for node in nodos}

    for i in range(n - 1):
        ni = nodos[i]
        nj = nodos[i + 1]
        xi, yi = ni.x, ni.y
        xj, yj = nj.x, nj.y

        dx = xj - xi
        dy = yj - yi
        L = longitudes[i]

        # Dirección normalizada del tramo (unitario)
        vx = dx / L
        vy = dy / L

        # Vector perpendicular (hacia abajo)
        nx = -vy
        ny = vx

        # Fuerza total sobre el tramo
        Fi = q_lineal * L  # Total fuerza sobre el tramo

        # Componente de fuerza en x y y (globales)
        fx = Fi * nx
        fy = Fi * ny

        # Distribuir mitad a cada nodo
        nodal_forces[ni.id] += np.array([fx / 2, fy / 2])
        nodal_forces[nj.id] += np.array([fx / 2, fy / 2])

    # Paso 5: aplicar fuerzas a la estructura
    for node in nodos:
        fx, fy = nodal_forces[node.id]
        dof_x, dof_y = node.dofs
        estructura.apply_force(dof_x, fx)
        estructura.apply_force(dof_y, fy)
        print(f"Nodo {node.id} ← Fx = {fx:.3f} N, Fy = {fy:.3f} N")


def apply_self_weight(elements, rho, estructure):

    for element in elements:
        centroid = element.get_centroid()

        area = element.area / 100**2
        espesor = element.section.thickness / 100
        peso = area * espesor * rho * 9.81 

        #Agrego la fuerza al elemento
        F_interna = element.apply_point_body_force(x=centroid[0], y=centroid[1],force_vector=[0, peso])

        for i in range(len(F_interna)):
            F_interna[i] =abs(F_interna[i])*-1

        node_a = element.node_list[0]
        node_b = element.node_list[1]
        node_c = element.node_list[2]

        dof_a = node_a.id * 2
        dof_b = node_b.id * 2
        dof_c = node_c.id * 2

        #Ahora agrego las cargas de peso propio

        estructure.apply_force(dof_index=dof_a, value=F_interna[1])
        estructure.apply_force(dof_index=dof_b, value=F_interna[3])
        estructure.apply_force(dof_index=dof_c, value=F_interna[5])
    
def compute_nodal_von_mises(elements, u_global):
        """
        Promedia los esfuerzos de Von Mises en los nodos a partir de los elementos vecinos.
        """
        nodal_vm = {}  # node.id : [list of vm from attached elements]

        for elem in elements:
            vm = elem.von_mises_stress(u_global)
            for node in elem.node_list:
                if node.id not in nodal_vm:
                    nodal_vm[node.id] = []
                nodal_vm[node.id].append(vm)

        # Promedio por nodo
        nodal_vm_avg = {node_id: np.mean(vms) for node_id, vms in nodal_vm.items()}
        return nodal_vm_avg

# ===============
# Función principal
# ===============
def main(title, self_weight=False, point_force=False, distribuited_force = False, def_scale=1, force_scale=1e-2, reaction_scale=1e-2):
    input_file = "ENTREGA_6/llave.geo"
    output_file = "ENTREGA_6/malla.msh"
    lc = 10

    q = 294 #N

    generate_mesh(input_file, output_file, lc)
    grupos, mesh = make_nodes_groups(output_file, title)
    sections, nodes_dict = make_sections(grupos)
    elements, nodes = make_cst_elements(mesh, sections, nodes_dict)

    plot_all_elements(elements, title)

    estructure = Solve(nodes, elements)

    if point_force:
        nodos_fuerza = grupos["Fuerza"]

        nodo = ''
        for i in range(len(nodos_fuerza)):
            if i == 0:
                nodo = nodos_fuerza[i]

            else:
                if nodos_fuerza[i].x > nodo.x:
                    nodo = nodos_fuerza[i]

        #Aplico una fuerza a ese nodo
        dof = nodo.id * 2
        estructure.apply_force(dof, -q)

    if distribuited_force:
        nodos_fuerza = grupos["Fuerza"]
        apply_distributed_force(nodos_fuerza, fuerza_total_y=q, estructura=estructure)

    # Aplicar peso propio
    rho = 1.2 #densidad

    if self_weight:
        # Aplicar peso propio a los elementos
        apply_self_weight(elements, rho, estructure)

    f = estructure.f_original if hasattr(estructure, 'f_original') else estructure.f_global

    plot_applied_forces(estructure.nodes, elements,title, f, scale=force_scale)

    desplazamientos = estructure.solve()

    # Importante: guardar los desplazamientos en cada nodo
    for node in estructure.nodes:
        node.structure = estructure  # para acceder a u_global desde cada nodo

    # Luego graficar
    plot_deformed_structure(estructure.elements, title, scale=def_scale, show_ids=False)

    reacciones = estructure.compute_reactions()

    plot_deformed_with_reactions(title, estructure.elements, reacciones, scale=def_scale, reaction_scale=reaction_scale, show_ids=False)

    vm_nodal = compute_nodal_von_mises(estructure.elements, estructure.u_global)
    plot_von_mises_field(estructure.nodes, estructure.elements, vm_nodal, title)
    
    def compute_nodal_stress_strain(nodes, elements, u_global):
        import numpy as np

        """
        Calcula σxx, σyy, σxy, εxx, εyy, εxy por nodo (promedio de elementos conectados).
        Retorna: diccionario {node.id: (σ_vec, ε_vec)}
        """
        stress_map = {node.id: [] for node in nodes}
        strain_map = {node.id: [] for node in nodes}

        for elem in elements:
            stress = elem.get_stress(u_global)
            strain = elem.get_strain(u_global)
            for node in elem.node_list:
                stress_map[node.id].append(stress)
                strain_map[node.id].append(strain)

        result = {}
        for node in nodes:
            σ_avg = np.mean(stress_map[node.id], axis=0) if stress_map[node.id] else np.zeros(3)
            ε_avg = np.mean(strain_map[node.id], axis=0) if strain_map[node.id] else np.zeros(3)
            result[node.id] = (σ_avg, ε_avg)

        return result  # node.id: (σ_vec, ε_vec)
    
    nodal_fields = compute_nodal_stress_strain(estructure.nodes, estructure.elements, estructure.u_global)

    for node_id, (sigma, epsilon) in nodal_fields.items():
        σxx, σyy, τxy = sigma
        εxx, εyy, γxy = epsilon
        print(f"Nodo {node_id}:")
        print(f"  σxx = {σxx:.2f}, σyy = {σyy:.2f}, τxy = {τxy:.2f}")
        print(f"  εxx = {εxx:.5e}, εyy = {εyy:.5e}, γxy = {γxy:.5e}")

    

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np
    import re
    import os

    def clean_filename(text):
        """Convierte un string en nombre de archivo válido eliminando LaTeX y símbolos especiales."""
        return re.sub(r'[^\w\-_. ]', '', text.replace('$', '').replace('\\', '').replace('{', '').replace('}', ''))

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
        
    nodal_fields = compute_nodal_stress_strain(estructure.nodes, estructure.elements, estructure.u_global)
    plot_all_scalar_fields_separately(estructure.nodes, estructure.elements, nodal_fields, title)

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
    
    def plot_principal_fields(nodes, elements, u_global, title_prefix=title):
        """
        Genera 4 gráficos: σ1, σ2, ε1, ε2.
        """
        sigma1, sigma2, eps1, eps2 = compute_nodal_principal_fields(nodes, elements, u_global)

        plot_scalar_field(nodes, elements, sigma1, r"$\sigma_1$ (Pa)", f"{title_prefix} - sigma_1")
        plot_scalar_field(nodes, elements, sigma2, r"$\sigma_2$ (Pa)", f"{title_prefix} - sigma_2")
        plot_scalar_field(nodes, elements, eps1, r"$\varepsilon_1$", f"{title_prefix} - epsilon_1")
        plot_scalar_field(nodes, elements, eps2, r"$\varepsilon_2$", f"{title_prefix} - epsilon_2")

    plot_principal_fields(estructure.nodes, estructure.elements, estructure.u_global, title_prefix=title)

if __name__ == "__main__":
    #title = 'Case a'
    #main(title, self_weight=False, point_force=True, distribuited_force = False, def_scale = 0.1, force_scale=0.05, reaction_scale = 1.5e-2)

    #title = 'Case b'
    #main(title, self_weight=False, point_force=False, distribuited_force = True, def_scale = 0.1, force_scale=0.2, reaction_scale = 1.5e-2)

    #title = 'Case c'
    #main(title, self_weight=True,  point_force=False, distribuited_force = True, def_scale = 0.1, force_scale=0.2, reaction_scale = 1.5e-2)

    #title = 'Case d'
    #main(title, self_weight=True,  point_force=False, distribuited_force = False, def_scale = 1000, force_scale=10000, reaction_scale = 100)

    title = 'Initial'
    main(title, self_weight=True,  point_force=False, distribuited_force = False, def_scale = 1000, force_scale=10000, reaction_scale = 100)
