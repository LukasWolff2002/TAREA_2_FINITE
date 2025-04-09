from nodes import Node
import numpy as np
import matplotlib.pyplot as plt
from section import Section
from cst import CST
from solve import Solve
from graph import plot_deformed_structure, plot_nodes, plot_sections
# ------------------------


b = 30 #mm
h = 30 #mm
L = 100 #mm

#Defino el tamaño de cada elemento en x e y
delta_x = 10 #mm
delta_y = 10 #mm

nodos_y = int(h/delta_y) + 1
nodos_x = int(L/delta_x) + 1

E = 210000


section = Section(thickness=1, E=E, nu=0.3)

#Por lo tanto, ahora defino nodos en el dominio (h,L)
nodes = []

node_id = 1
for i in range(0, int(L/delta_x)+1):
    for j in range(0, int(h/delta_y)+1):
        dof_id = 2 * (node_id - 1)
        if i == 0:
            nodes.append(Node(node_id, i*delta_x, j*delta_y, [dof_id, dof_id + 1], restrain=[1, 1]))
        else:
            nodes.append(Node(node_id, i*delta_x, j*delta_y, [dof_id, dof_id + 1]))
        node_id += 1

#Ahora ensabmlo los elementos
elements = []
i = 1
for node in nodes:

    if node.x == L:
        #Detengo el loop cuando llego al a la linea final de elementos
        #Ya que no debo crear mas CST
        break

    if i%nodos_y != 0:
        #Debo conectar esta nodo  con dos nodos de la derecha
        nodo_a = node
        nodo_b = nodes[i+nodos_y-1]
        nodo_c = nodes[i+nodos_y]

        #Creo el elemento
        elements.append(CST(i, [nodo_a, nodo_b, nodo_c], section))

        nodo_d = nodes[i]
        nodo_e = nodes[i+nodos_y]

        #Creo el elemento
        elements.append(CST(i+1, [nodo_a, nodo_e, nodo_d], section, color='red'))

   
        

    i += 1

#Plotear los nodos
#plot_nodes(nodes)

#Plotear los elementos
#plot_sections(elements)

    


# Crear la estructura
estructura = Solve(nodes, elements)

nodo_objetivo = nodes[-1]  # último nodo
ux_dof = nodo_objetivo.dofs[0]  # DOF en x
uy_dof = nodo_objetivo.dofs[1]  # DOF en y

#dof id es el ultimo dof en vertical
estructura.apply_force(dof_index=ux_dof, value=1000)   # carga horizontal
estructura.apply_force(dof_index=uy_dof, value=-2000)  # carga vertical hacia abajo
desplazamientos = estructura.solve()


plot_deformed_structure(nodes, elements, estructura.u_global, scale=1)




