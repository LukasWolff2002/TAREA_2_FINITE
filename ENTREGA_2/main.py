from nodes import Node
import numpy as np
import matplotlib.pyplot as plt
from section import Section
from cst import CST
from solve import Solve
from graph import plot_full_structure
from assembly import Structure
# ------------------------

def Run_script (b,h,L,delta_x, delta_y, Px=0, Force_x=False):

    E = 210000 #GPa

    structure_1 = Structure(b, h, L, delta_x, delta_y, E)

    dof_x = []
    dof_y = []
    for node in structure_1.nodes:
        if node.x == L:
            dof_x.append(node.dofs[0])
            dof_y.append(node.dofs[1])

    # Crear la estructura
    estructura = Solve(structure_1.nodes, structure_1.elements)

    # Carga por peso propio
    rho = 7850e-9  # N/mm³
    g = 9.81       # m/s²
    f_body = np.array([0, -rho * g])

    # Aplicar a todos los elementos
    for elem in estructura.elements:
        f_eq = elem.body_forces(f_body)
        dofs = elem.calculate_indices()

        for i, dof in enumerate(dofs):
            estructura.f_global[dof] += f_eq[i]

    if Force_x:
        p = Px/len(dof_x)
        for dof in dof_x:
            estructura.apply_force(dof_index=dof, value=-p)  # Carga horizontal

    # Resolver
    desplazamientos = estructura.solve()

    nodo_des = structure_1.nodes[-1]

    ux_dof = nodo_des.dofs[0]  # índice del DOF en x
    uy_dof = nodo_des.dofs[1]  # índice del DOF en y

    ux = estructura.u_global[ux_dof, 0]
    uy = estructura.u_global[uy_dof, 0]

    print(f"Desplazamiento del nodo extremo: ux = {ux:.6e} mm, uy = {uy:.6e} mm")

    # Calcular tensiones en todos los elementos
    tensiones = []

    for elem in estructura.elements:
        stress = elem.get_stress(estructura.u_global)
        tensiones.append((elem.element_tag, elem.get_centroid(), stress))

    plot_full_structure(structure_1.nodes, structure_1.elements, u_global=estructura.u_global, deform_scale=1000)
