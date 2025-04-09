from nodes import Node
import numpy as np
import matplotlib.pyplot as plt
from section import Section
from cst import CST
from solve import Solve
from graph import plot_full_structure
from assembly import Structure
# ------------------------

b = 200 #mm
h = 400 #mm
L = 3000 #mm

#Defino el tamaño de cada elemento en x e y
delta_x = 50 #mm
delta_y = 50 #mm

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


# Resolver
desplazamientos = estructura.solve()
plot_full_structure(structure_1.nodes, structure_1.elements, u_global=estructura.u_global, deform_scale=1000)

