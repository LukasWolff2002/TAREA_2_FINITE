import os
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio
import time

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
def main(lc, title, self_weight=False, point_force=False, distribuited_force = False, def_scale=1, force_scale=1e-2, reaction_scale=1e-2):
    input_file = "ENTREGA_6/llave.geo"
    output_file = "ENTREGA_6/malla.msh"

    q = 294 #N

    generate_mesh(input_file, output_file, lc)
    grupos, mesh = make_nodes_groups(output_file, title)
    sections, nodes_dict = make_sections(grupos)
    elements, nodes = make_cst_elements(mesh, sections, nodes_dict)

    

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
        apply_self_weight(elements, rho, estructure)

    f = estructure.f_original if hasattr(estructure, 'f_original') else estructure.f_global

    start_time = time.time()  # Guarda el tiempo inicial
    estructure.solve()
    end_time = time.time()  # Guarda el tiempo después de ejecutar
    elapsed_time = end_time - start_time  # Calcula la diferencia

    # Importante: guardar los desplazamientos en cada nodo
    for node in estructure.nodes:
        node.structure = estructure  # para acceder a u_global desde cada nodo

    # Luego graficar
    


    vm_nodal = compute_nodal_von_mises(estructure.elements, estructure.u_global)
    vm_nodal = np.array(list(vm_nodal.values()))


    
    
    return vm_nodal.max(), elapsed_time
    
    
    

if __name__ == "__main__":
    title = 'Convergencia'

    LC = [40, 35, 30, 25, 20, 15, 10, 8, 6, 4, 2, 1, 0.8, 0.6]
    #LC = [40, 35, 30, 2]
    Von_mises = []
    von_mises = []
    Tiempo = []

    for lc in LC:
        resultados = main(lc, title, self_weight=True, distribuited_force=True, def_scale=1, force_scale=1e-2, reaction_scale=1e-2)
        vm = resultados[0]
        tiempo = resultados[1]
        Von_mises.append(vm)
        Tiempo.append(tiempo)

    
    for lc, vm in zip(LC, Von_mises):
        von_mises.append(vm)

    print(von_mises)
    print(Tiempo)



    fig, ax1 = plt.subplots()

    # Primer eje Y: Von Mises
    ax1.set_xlabel('Longitud Característica (LC)')
    ax1.set_ylabel('Tensión de Von Mises (MPa)', color='tab:red')
    ax1.set_xscale('log')  # Escala logarítmica en eje X
    ax1.plot(LC, von_mises, color='tab:red', marker='o', label='Von Mises')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Segundo eje Y: Tiempo
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tiempo de ejecución (s)', color='tab:blue')
    ax2.plot(LC, Tiempo, color='tab:blue', marker='s', label='Tiempo')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Convergencia de Von Mises y Tiempo vs LC')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig('INFORME/GRAFICOS/convergencia.png', dpi=300)
