from nodes import Node
import numpy as np
import matplotlib.pyplot as plt
from section import Section
from cst import CST
from solve import Solve
from graph import plot_full_structure
from assembly import Structure
# ------------------------

b = 30 #mm
h = 30 #mm
L = 100 #mm

#Defino el tamaño de cada elemento en x e y
delta_x = 10 #mm
delta_y = 10 #mm

structure_1 = Structure(b, h, L, delta_x, delta_y)



    


# Crear la estructura
estructura = Solve(structure_1.nodes, structure_1.elements)

nodo_objetivo = structure_1.nodes[-1]  # último nodo
ux_dof = nodo_objetivo.dofs[0]  # DOF en x
uy_dof = nodo_objetivo.dofs[1]  # DOF en y

#dof id es el ultimo dof en vertical
estructura.apply_force(dof_index=ux_dof, value=100000)   # carga horizontal
#estructura.apply_force(dof_index=uy_dof, value=-2000)  # carga vertical hacia abajo
desplazamientos = estructura.solve()

plot_full_structure(structure_1.nodes, structure_1.elements, u_global=estructura.u_global)

