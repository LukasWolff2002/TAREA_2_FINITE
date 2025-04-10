from nodes import Node
from cst import CST
from section import Section
import numpy as np
#from solve import ensamblar_y_resolver

# ------------------------
# Caso de estudio (1 elemento) Ejemplo en 15.3.6 y Ejercicio 15.5 Felippa
# ------------------------

# Nodos
node1 = Node(1, 0.0, 0.0, [0, 1], restrain=[1, 1]) #Restricciones para no hacerlo mecanismo
node2 = Node(2, 3.0, 1.0, [2, 3], restrain=[0, 0])
node3 = Node(3, 2.0, 2.0, [4, 5], restrain=[0, 0])
nodes = [node1, node2, node3]

# Sección
E = np.array([[100,25,0],[25,100,0],[0,0,50]]) # 8*np.array([[4,1,0], [1,4,0],[0,0,2]]) #
t = 1 # Espesor de la sección
section = Section(thickness=t, E=E, nu=0.3)

# Elemento CST
element = CST(1, [node1, node2, node3], section)

# Imprimir resumen
element.printSummary()
# Aplicar una fuerza puntual como carga interna en punto (2.0, 1.5) cualquiera 
F_interna = element.apply_point_body_force(x=2.0, y=1.5,force_vector=[0, -1000], use_centroid=True)

# Resolver con esta fuerza
u, f, K = element.ensamblar_y_resolver(F_interna, nodes)

u.flatten(), f.flatten(), K

# Mostrar desplazamientos por nodo
print("\nDesplazamientos por nodo:")
for node in nodes:
    ux = u[node.dofs[0]][0]
    uy = u[node.dofs[1]][0]
    print(f"Nodo {node.id}: ux = {ux:.4e} m, uy = {uy:.4e} m")

"""
Los desplazamientos son de gran magnitud porque solo se tiene un nodo restringido, la fuerza es muy grande y el 
módulo de elasticidad es muy bajo.
"""


element.plotGeometry()
