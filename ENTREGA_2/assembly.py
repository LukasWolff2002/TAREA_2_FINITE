import numpy as np
from section import Section
from nodes import Node
from cst import CST









class Structure:
    def __init__(self, b, h, L, delta_x, delta_y, E=210000):
        self.b = b
        self.h = h
        self.L = L
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.nodos_y = int(h / delta_y) + 1
        self.nodos_x = int(L / delta_x) + 1
        self.section = Section(thickness=1, E=E, nu=0.3)

        self.nodes = []
        self.elements = []

        self._create_nodes()
        self._create_elements()

    def _create_nodes(self):
        node_id = 1
        for i in range(self.nodos_x):
            for j in range(self.nodos_y):
                x = i * self.delta_x
                y = j * self.delta_y
                dof_id = 2 * (node_id - 1)
                restrain = [1, 1] if i == 0 else [0, 0]  # Fijar nodos en el borde izquierdo
                self.nodes.append(Node(node_id, x, y, [dof_id, dof_id + 1], restrain=restrain))
                node_id += 1

    def _create_elements(self):
        for i, node in enumerate(self.nodes):
            if node.x == self.L:
                continue  # saltar nodos en el borde derecho

            if (i + 1) % self.nodos_y != 0:
                # Nodo base de dos triángulos
                nodo_a = node
                nodo_b = self.nodes[i + self.nodos_y]
                nodo_c = self.nodes[i + self.nodos_y + 1]
                nodo_d = self.nodes[i + 1]

                # Triángulo inferior (base a la izquierda)
                self.elements.append(CST(len(self.elements) + 1, [nodo_a, nodo_b, nodo_c], self.section))

                # Triángulo superior (base a la derecha)
                self.elements.append(CST(len(self.elements) + 1, [nodo_a, nodo_c, nodo_d], self.section, color='red'))
