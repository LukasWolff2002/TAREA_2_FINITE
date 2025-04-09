from nodes import Node
import numpy as np
import matplotlib.pyplot as plt
from section import Section
from cst import CST
from solve import ensamblar_y_resolver

b = 30 #mm
h = 30 #mm
L = 100 #mm

#Defino el tamaño de cada elemento en x e y
delta_x = 10 #mm
delta_y = 10 #mm

nodos_y = int(h/delta_y) + 1
nodos_x = int(L/delta_x) + 1

section = Section(thickness=1, E=np.array([[4,1,0], [1,4,0],[0,0,2]]), nu=0.3)

print(f"Total de nodos en x: {nodos_x}, Total de nodos en y: {nodos_y}")

def plot_nodes(nodes):
    plt.figure(figsize=(8, 4))
    for node in nodes:
        plt.plot(node.x, node.y, 'bo')  # nodo como punto azul
        plt.text(node.x + 1, node.y + 1, str(node.id), fontsize=9, color='red')  # ID al lado del nodo

    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.title('Distribución de Nodos')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

#Por lo tanto, ahora defino nodos en el dominio (h,L)
nodes = []

node_id = 1
for i in range(0, int(L/delta_x)+1):
    for j in range(0, int(h/delta_y)+1):
        nodes.append(Node(node_id, i*delta_x, j*delta_y, [2*i+j, 2*i+j+1]))
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
plot_nodes(nodes)
def plot_sections(elements, alpha=0.5):
    plt.figure(figsize=(10, 5))

    for elem in elements:
        coords = elem.get_xy_matrix()
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Relleno del triángulo con color y opacidad
        plt.fill(x, y, color=elem.color, alpha=alpha, edgecolor='k')

        # Trazar borde del triángulo
        coords_closed = np.vstack([coords, coords[0]])  # cerrar triángulo
        plt.plot(coords_closed[:, 0], coords_closed[:, 1], 'k-', linewidth=0.5)

        # Mostrar ID del elemento en el centroide
        centroid = elem.get_centroid()
        plt.text(centroid[0], centroid[1], f'E{elem.element_tag}', fontsize=8, color='red')

        # Mostrar nodos
        for node in elem.node_list:
            plt.plot(node.x, node.y, 'bo')
            plt.text(node.x + 0.5, node.y + 0.5, f'N{node.id}', fontsize=7, color='blue')

    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Elementos CST con color")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Plotear los elementos
plot_sections(elements)



