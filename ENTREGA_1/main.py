from nodes import Node
from cst import CST
from section import Section
import numpy as np
from solve import ensamblar_y_resolver

# ------------------------
# Caso de estudio (1 elemento)
# ------------------------

# Nodos
node1 = Node(1, 0.0, 0.0, [0, 1], restrain=['r', 'r'])
node2 = Node(2, 3.0, 1.0, [2, 3], restrain=['f', 'f'])
node3 = Node(3, 2.0, 2.0, [4, 5], restrain=['f', 'f'])
nodes = [node1, node2, node3]

# Sección
E = 8*np.array([[4,1,0], [1,4,0],[0,0,2]])
section = Section(thickness=1, E=E, nu=0.3)

# Elemento CST
element = CST(1, [node1, node2, node3], section)

# Imprimir resumen
element.printSummary()
element.body_forces([0, -1000])
element.plotGeometry()
# Aplicar carga puntual en nodo
F_nodal = element.get_point_load_forces(node3.x, node3.y, [0, -1000])

# Resolver sistema
u, f, K = ensamblar_y_resolver(element, F_nodal, nodes)

u.flatten(), f.flatten(), K
