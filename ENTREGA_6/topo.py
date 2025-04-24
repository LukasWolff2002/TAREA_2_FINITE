import os
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import re
import os


from nodes import Node
from section import Section
from cst import CST
from solve import Solve

from graph import plot_all_elements, plot_applied_forces, plot_deformed_structure, plot_deformed_with_reactions, plot_von_mises_field, plot_all_scalar_fields_separately, plot_principal_fields, plot_elements_by_thickness

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
    #Node.plot_nodes_por_grupo(grupos, title, show_ids=False)

    return grupos, mesh

# ========================================
# Crear secciones del modelo y asociarlas
# ========================================
def make_sections(grupos):
    thickness = {"1": 1.6, "2": 3.0, "3": 5.0, "4": 5.0}
    sections = {}

    # Propiedades del material (ortotrópico PLA impreso)
    for group in thickness:
        sections[group] = Section(thickness[group], E=3500, nu=0.36)

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
        #print(f"Nodo {node.id} ← Fx = {fx:.3f} N, Fy = {fy:.3f} N")

import meshio

def export_structure_to_3d_stl_symmetric(elements, filename="modelo_optimizado_3d_sym.stl"):
    """
    Exporta un STL 3D donde cada triángulo se extruye simétricamente ±espesor/2 respecto al plano z=0.
    El resultado es un volumen sólido y centrado.
    """
    import meshio

    vertices = []
    cells = []

    for elem in elements:
        coords = elem.get_xy_matrix()
        t = elem.section.thickness   # mm → m
        z_bot = -t / 2
        z_top = +t / 2

        # Crear vértices
        base_idx = []
        top_idx = []
        for (x, y) in coords:
            idx_bot = len(vertices)
            vertices.append([x, y, z_bot])
            base_idx.append(idx_bot)

            idx_top = len(vertices)
            vertices.append([x, y, z_top])
            top_idx.append(idx_top)

        # Cara inferior (sentido antihorario visto desde abajo)
        cells.append([base_idx[0], base_idx[1], base_idx[2]])

        # Cara superior (reversa)
        cells.append([top_idx[2], top_idx[1], top_idx[0]])

        # Caras laterales
        for i in range(3):
            j = (i + 1) % 3
            v0 = base_idx[i]
            v1 = base_idx[j]
            v2 = top_idx[j]
            v3 = top_idx[i]
            cells.append([v0, v1, v2])
            cells.append([v0, v2, v3])

    # Crear y exportar STL
    mesh = meshio.Mesh(
        points=np.array(vertices),
        cells=[("triangle", np.array(cells))]
    )
    mesh.write(filename)
    print(f"✅ STL 3D simétrico exportado: {filename}")

def export_structure_to_3d_stl_symmetric_smoothed(elements, filename="modelo_optimizado_3d_sym_suavizado.stl", smooth_steps=10, alpha=0.5):
    """
    Exporta un STL 3D extruyendo los triángulos con espesor ±t/2 y suaviza la malla (sin cambiar conectividad).
    Aplica suavizado Laplaciano a los vértices para redondear visualmente el cuerpo.
    """
    import numpy as np
    import meshio
    from collections import defaultdict

    vertices = []
    cells = []
    vertex_map = {}  # (x, y) → index
    neighbors = defaultdict(set)

    for elem in elements:
        coords = elem.get_xy_matrix()
        t = elem.section.thickness
        z_bot = -t / 2
        z_top = +t / 2

        local_indices = []

        for (x, y) in coords:
            key = (x, y)
            if key not in vertex_map:
                vertex_map[key] = len(vertices) // 2
                vertices.append([x, y, z_bot])
                vertices.append([x, y, z_top])
            local_indices.append(vertex_map[key])

        a, b, c = local_indices
        bot = [2*a, 2*b, 2*c]
        top = [2*c+1, 2*b+1, 2*a+1]  # reversed

        cells.append(bot)
        cells.append(top)

        # Caras laterales
        for i in range(3):
            j = (i + 1) % 3
            v0 = 2 * local_indices[i]
            v1 = 2 * local_indices[j]
            v2 = 2 * local_indices[j] + 1
            v3 = 2 * local_indices[i] + 1
            cells.append([v0, v1, v2])
            cells.append([v0, v2, v3])

            # Guardar vecinos en 2D
            neighbors[local_indices[i]].add(local_indices[j])
            neighbors[local_indices[j]].add(local_indices[i])

    # Suavizado Laplaciano sobre XY (manteniendo Z fija por plano)
    verts_array = np.array(vertices)
    for _ in range(smooth_steps):
        new_xy = verts_array[:, :2].copy()
        for idx in vertex_map.values():
            nbrs = list(neighbors[idx])
            if not nbrs:
                continue
            avg = np.mean([verts_array[2*n][:2] for n in nbrs], axis=0)
            new_xy[2*idx] = (1 - alpha) * new_xy[2*idx] + alpha * avg
            new_xy[2*idx+1] = (1 - alpha) * new_xy[2*idx+1] + alpha * avg
        verts_array[:, :2] = new_xy

    # Exportar STL
    mesh = meshio.Mesh(
        points=verts_array,
        cells=[("triangle", np.array(cells))]
    )
    mesh.write(filename)
    print(f"✅ STL 3D suavizado exportado: {filename}")






def apply_self_weight(elements, rho, estructure):
    P = 0
    for element in elements:
        centroid = element.get_centroid()

        area = element.area / 100**2
        espesor = element.section.thickness / 100
        peso = area * espesor * rho * 9.81 
        P += peso

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

    print(f"Peso total de la estructura: {P:.5f} N")
    return P
    
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

def compute_nodal_stress_strain(nodes, elements, u_global):

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

def optimize_topology_iterative_n_extremes(P, grupos, elements, nodes, rho, estructure,
                                           num_iterations=5, num_elements=2,
                                           delta_t=0.2, t_min=0.2, t_max=10.0):
    """
    Optimización topológica iterativa con propagación ultra-suavizada:
    - Aplica cambios principales a los N elementos extremos
    - Propaga ajustes suaves a través de una función Gaussiana acumulativa
    """
    import numpy as np
    import math
    from collections import defaultdict

    q = 294
    g = 9.81

    def gaussian_weight(level, sigma=2.0):
        return math.exp(-0.5 * (level / sigma) ** 2)

    def find_neighbors_recursive(start_indices, levels):
        neighbor_levels = defaultdict(set)
        current = set(start_indices)
        visited = set(start_indices)

        for level in range(1, levels + 1):
            next_neighbors = set()
            target_nodes = set(n for idx in current for n in elements[idx].node_list)

            for i, elem in enumerate(elements):
                if i in visited:
                    continue
                if any(n in target_nodes for n in elem.node_list):
                    neighbor_levels[level].add(i)
                    next_neighbors.add(i)

            visited.update(next_neighbors)
            current = next_neighbors

        return neighbor_levels

    def update_element_thickness(elem, delta, tag):
        t_old = elem.section.thickness
        t_new = np.clip(t_old + delta, t_min, t_max)
        elem.section = Section(t_new, E=3500, nu=0.36)
        elem.Ke = elem.get_stiffness_matrix()
        updated_indices.add(elem.element_tag)
        #print(f"{tag} → Elem {elem.element_tag} | t: {t_old:.3f} → {t_new:.3f}")

    for it in range(num_iterations):
        print(f"\n🔁 Iteración {it+1}/{num_iterations}")
        print(f"El peso original es: {P:.5f} N")

        estructure = Solve(nodes, elements)
        apply_self_weight(elements, rho, estructure)
        apply_distributed_force(grupos["Fuerza"], fuerza_total_y=q, estructura=estructure)
        estructure.solve()

        for node in estructure.nodes:
            node.structure = estructure

        von_mises = np.array([elem.von_mises_stress(estructure.u_global) for elem in elements])
        sorted_indices = np.argsort(von_mises)

        max_indices = sorted_indices[-num_elements:]
        min_indices = sorted_indices[:num_elements]

        updated_indices = set()

        # Aplicar cambio principal
        for idx in max_indices:
            update_element_thickness(elements[idx], +delta_t, "🔺 max")

        for idx in min_indices:
            update_element_thickness(elements[idx], -delta_t, "🔻 min")

        # Propagación ultra-suavizada
        sigma = 2.0
        levels = 6  # hasta vecinos de 6º orden

        max_neighbors_by_level = find_neighbors_recursive(max_indices, levels)
        min_neighbors_by_level = find_neighbors_recursive(min_indices, levels)

        for level in range(1, levels + 1):
            weight = gaussian_weight(level, sigma) * delta_t
            for idx in max_neighbors_by_level[level]:
                if elements[idx].element_tag in updated_indices:
                    continue
                update_element_thickness(elements[idx], +weight, f"⤴ nivel {level}")
            for idx in min_neighbors_by_level[level]:
                if elements[idx].element_tag in updated_indices:
                    continue
                update_element_thickness(elements[idx], -weight, f"⤵ nivel {level}")

        # Reportar peso
        peso_total = sum(
            (el.area / 100**2) * (el.section.thickness / 100) * rho * g
            for el in elements
        )
        print(f"⚖️ Peso total aproximado: {peso_total:.5f} N")

                # Suavizar reducción de masa si el peso excede
        if peso_total > P:
            exceso = peso_total - P
            print(f"❌ Exceso de masa: {exceso:.3f} N — se reducirá espesor suavemente")

            sigma_red = 1.0
            levels_red = 6
            reduction_step = delta_t / 2
            weight_by_level = {
                level: gaussian_weight(level, sigma_red) * reduction_step
                for level in range(1, levels_red + 1)
            }

            still_exceeds = True
            temp_updated = set(updated_indices)

            max_mass_reduction_steps = 100

            all_reducible_candidates = [
                    (i, von_mises[i])
                    for i in range(len(elements))
                    if elements[i].section.thickness > t_min+0.2 and i not in temp_updated
                ]

            for reduction_iter in range(max_mass_reduction_steps):
                print(f"\n🔁 Paso de reducción #{reduction_iter + 1} — Peso actual: {peso_total:.3f} N")

                # 🔁 Recalcular esfuerzos actualizados después de cambios de espesor
                estructure = Solve(nodes, elements)
                apply_self_weight(elements, rho, estructure)
                apply_distributed_force(grupos["Fuerza"], fuerza_total_y=q, estructura=estructure)
                estructure.solve()
                von_mises = np.array([elem.von_mises_stress(estructure.u_global) for elem in elements])

                # Ordenar todos los candidatos posibles por von Mises (de menor a mayor)
                
                sorted_reduction_indices = [i for i, _ in sorted(all_reducible_candidates, key=lambda x: x[1])]

                print(f"   ➤ Candidatos a reducir: {len(sorted_reduction_indices)}")

                # Buscar `num_elements` candidatos con al menos 0.2 mm de margen
                base_indices = []
                for i in sorted_reduction_indices:
                    if elements[i].section.thickness >= t_min + 1:
                        base_indices.append(i)
                    if len(base_indices) == num_elements:
                        break

                # Si no hay suficientes con margen, completar con otros que tengan t > t_min
                if len(base_indices) < num_elements:
                    for i in sorted_reduction_indices:
                        if i in base_indices:
                            continue
                        if elements[i].section.thickness > t_min+1:
                            base_indices.append(i)
                        if len(base_indices) == num_elements:
                            break

                # Verificación final
                if not base_indices:
                    print("⚠️ No quedan elementos con espesor suficiente para seguir reduciendo masa.")
                    break

                # Aplicar reducción
                for base_idx in base_indices:
                    elem = elements[base_idx]
                    t_old = elem.section.thickness
                    t_new = max(t_old - reduction_step, t_min)
                    if t_new < t_old:
                        elem.section = Section(t_new, E=3500, nu=0.36)
                        elem.Ke = elem.get_stiffness_matrix()
                        peso_total -= (elem.area / 100**2) * ((t_old - t_new) / 100) * rho * g
                        temp_updated.add(base_idx)

                    # Reducir vecinos suavemente
                    neighbors_by_level = find_neighbors_recursive([base_idx], levels_red)
                    for level, idxs in neighbors_by_level.items():
                        for idx in idxs:
                            if idx in temp_updated:
                                continue
                            elem_n = elements[idx]
                            if elem_n.section.thickness <= t_min:
                                continue
                            t_old_n = elem_n.section.thickness
                            t_new_n = max(t_old_n - weight_by_level[level], t_min)
                            if t_new_n < t_old_n:
                                elem_n.section = Section(t_new_n, E=3500, nu=0.36)
                                elem_n.Ke = elem_n.get_stiffness_matrix()
                                # Recalcular peso total exacto después de todos los cambios
                                peso_total = sum(
                                    (el.area / 100**2) * (el.section.thickness / 100) * rho * g
                                    for el in elements
                                )

                                temp_updated.add(idx)

                print(f"   ➤ Nuevo peso total: {peso_total:.5f} N")

                if peso_total <= P:
                    print(f"✅ Peso ajustado suavemente en {reduction_iter + 1} pasos.")
                    still_exceeds = False
                    break

            if still_exceeds:
                print("⚠️ No fue posible ajustar completamente el peso: límite mínimo de espesor alcanzado.")




    return estructure








# ===============
# Función principal
# ===============
def main(title, self_weight=False, point_force=False, distribuited_force = False, def_scale=1, force_scale=1e-2, reaction_scale=1e-2):
    input_file = "ENTREGA_6/llave.geo"
    output_file = "ENTREGA_6/malla.msh"
    lc = 2

    q = 294 #N

    generate_mesh(input_file, output_file, lc)
    grupos, mesh = make_nodes_groups(output_file, title)
    sections, nodes_dict = make_sections(grupos)
    elements, nodes = make_cst_elements(mesh, sections, nodes_dict)

    #plot_all_elements(elements, title)

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
    rho = 1.252 #densidad

    if self_weight:
        # Aplicar peso propio a los elementos
        P = apply_self_weight(elements, rho, estructure)

    f = estructure.f_original if hasattr(estructure, 'f_original') else estructure.f_global

    #plot_applied_forces(estructure.nodes, elements,title, f, scale=force_scale)

    desplazamientos = estructure.solve()

    # Guardar los desplazamientos en cada nodo
    for node in estructure.nodes:
        node.structure = estructure

    vm_nodal = compute_nodal_von_mises(estructure.elements, estructure.u_global)
    plot_von_mises_field(estructure.nodes, estructure.elements, vm_nodal, 'Topo_ini')


    #    optimize_topology_mass_constrained(estructure.elements, rho, delta=0.2)

    estructure = optimize_topology_iterative_n_extremes(P=P,
    grupos=grupos,
    elements=elements,
    nodes=nodes,
    rho=rho,
    estructure=estructure,
    num_iterations=50,
    num_elements=10,        # cambia este valor según qué tan agresiva sea la optimización
    delta_t=0.2,
    t_min=1,
    t_max=10.0
)



    plot_elements_by_thickness(estructure.elements)

    # Importante: guardar los desplazamientos en cada nodo
    for node in estructure.nodes:
        node.structure = estructure  # para acceder a u_global desde cada nodo


    vm_nodal = compute_nodal_von_mises(estructure.elements, estructure.u_global)
    plot_von_mises_field(estructure.nodes, estructure.elements, vm_nodal, title)
    
    nodal_fields = compute_nodal_stress_strain(estructure.nodes, estructure.elements, estructure.u_global)   

    export_structure_to_3d_stl_symmetric(elements, filename="ENTREGA_6/modelo_optimizado_3d_sym.stl")






if __name__ == "__main__":

    title = 'Topo'
    main(title, self_weight=True,  point_force=False, distribuited_force = True, def_scale = 1000, force_scale=10000, reaction_scale = 100)
