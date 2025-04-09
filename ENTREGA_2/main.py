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

for dof in dof_y:
    estructura.apply_force(dof_index=dof, value=-20000)  # carga vertical hacia abajo
 
desplazamientos = estructura.solve()

nodo_des = structure_1.nodes[-1]

ux_dof = nodo_des.dofs[0]  # índice del DOF en x
uy_dof = nodo_des.dofs[1]  # índice del DOF en y

ux = estructura.u_global[ux_dof, 0]
uy = estructura.u_global[uy_dof, 0]

print(f"Desplazamiento del nodo {nodo_des.id}: ux = {ux:.6e} mm, uy = {uy:.6e} mm")

plot_full_structure(structure_1.nodes, structure_1.elements, u_global=estructura.u_global, deform_scale=0.1)

