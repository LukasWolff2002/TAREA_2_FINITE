import numpy as np
import matplotlib.pyplot as plt

class CST:
    def __init__(self, element_tag, node_list, section, body_force=None, color='blue'):
        self.element_tag = element_tag
        self.node_list = node_list
        self.color = color
        self.section = section
        self.area = self.compute_area()
        self.B = self.get_B_matrix()
        self.Ke = self.get_stiffness_matrix()
        self.body_force = body_force
        if self.body_force is not None:
            self.body_forces(self.body_force)
        
    def calculate_indices(self):
        return np.concatenate([node.dofs for node in self.node_list])

    def get_xy_matrix(self):
        return np.array([[node.x, node.y] for node in self.node_list])

    def get_centroid(self):
        return np.mean(self.get_xy_matrix(), axis=0)

    def compute_area(self):
        x = [n.x for n in self.node_list]
        y = [n.y for n in self.node_list]
        return 0.5 * abs(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))

    def get_B_matrix(self):
        x = [n.x for n in self.node_list]
        y = [n.y for n in self.node_list]
        b = [y[1]-y[2], y[2]-y[0], y[0]-y[1]]
        c = [x[2]-x[1], x[0]-x[2], x[1]-x[0]]
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2*i] = b[i]
            B[1, 2*i+1] = c[i]
            B[2, 2*i] = c[i]
            B[2, 2*i+1] = b[i]
        return B / (2 * self.area)

    def get_stiffness_matrix(self):
        D = self.section.D
        t = self.section.thickness
        return t * self.area * self.B.T @ D @ self.B
    
    # Añadimos función para calcular e imprimir las fuerzas de cuerpo

    def body_forces(self, body_force_vector):
        """
        Calcula las fuerzas nodales equivalentes por una carga de cuerpo constante
        en el elemento CST.
        """
        t = self.section.thickness
        A = self.area
        bx, by = body_force_vector
        f_body = (t * A / 3) * np.array([bx, by, bx, by, bx, by])
        
        print("\nFuerzas nodales equivalentes por carga de cuerpo:")
        print(f"Vector de carga de cuerpo aplicado: {body_force_vector}")
        print("Fuerzas nodales equivalentes:")
        print(np.round(f_body, 2))
        
        return f_body


    def get_point_load_forces(self, x, y, force_vector):
        N = self.get_shape_functions_at(x, y)
        fx, fy = force_vector
        return (N.T @ np.array([fx, fy])).flatten()


    def get_shape_functions_at(self, x, y):
        x1, y1 = self.node_list[0].x, self.node_list[0].y
        x2, y2 = self.node_list[1].x, self.node_list[1].y
        x3, y3 = self.node_list[2].x, self.node_list[2].y
        A = self.area * 2

        N1 = ((x2*y3 - x3*y2) + (y2 - y3)*x + (x3 - x2)*y) / A
        N2 = ((x3*y1 - x1*y3) + (y3 - y1)*x + (x1 - x3)*y) / A
        N3 = ((x1*y2 - x2*y1) + (y1 - y2)*x + (x2 - x1)*y) / A

        return np.array([
            [N1, 0, N2, 0, N3, 0],
            [0, N1, 0, N2, 0, N3]
        ])
    

    def printSummary(self):
        print(f"\nCST Element {self.element_tag}")
        print("Type: planeStress")
        print("Nodes:")
        for n in self.node_list:
            print(f"  Node {n.id}: ({n.x}, {n.y}) DOFs: {n.dofs}")
        print(f"\nDOF indices: {self.calculate_indices()}")
        print(f"Area: {self.area:.4f}")
        print("\nXY matrix:\n", self.get_xy_matrix())
        print("Centroid:", self.get_centroid())
        print("\nMatrix B:\n", self.B)
        print("\nStiffness Matrix Ke:\n", self.Ke)
              

    def plotGeometry(self):
        coords = self.get_xy_matrix()
        coords = np.vstack([coords, coords[0]])
        plt.plot(coords[:, 0], coords[:, 1], 'k-o')
        for node in self.node_list:
            plt.text(node.x, node.y, f'N{node.id}', color='blue')
        centroid = self.get_centroid()
        plt.text(centroid[0], centroid[1], f'E{self.element_tag}', color='red')
        plt.axis('equal')
        plt.title(f'CST Element {self.element_tag}')
        plt.grid(True)
        plt.show()
