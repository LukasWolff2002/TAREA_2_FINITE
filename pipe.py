import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Clases auxiliares
# ------------------------

class Node:
    def __init__(self, id, x, y, dofs, restrain=None):
        self.id = id
        self.x = x
        self.y = y
        self.dofs = np.array(dofs)
        self.restrain = np.array(restrain if restrain else ['f', 'f'])

class Section:
    def __init__(self, thickness, E, nu, type='planeStress'):
        self.thickness = thickness
        self.E = E
        self.nu = nu
        self.type = type
        self.D = self._compute_D()

    def _compute_D(self):
        E, nu = self.E, self.nu
        
        if isinstance(E, np.ndarray):
            return E
        
        if self.type == 'planeStress':
            return (E / (1 - nu**2)) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ])
        elif self.type == 'planeStrain':
            coef = E / ((1 + nu)*(1 - 2*nu))
            return coef * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2*nu) / 2]
            ])
        else:
            raise ValueError(f"Invalid type: {self.type}")

# ------------------------
# Clase CST
# ------------------------

class CST:
    def __init__(self, element_tag, node_list, section, body_force=None):
        self.element_tag = element_tag
        self.node_list = node_list
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

# ------------------------
# Ensamblaje y resolución
# ------------------------

def ensamblar_y_resolver(elem, fuerza, nodos):
    total_dofs = 6
    K = np.zeros((total_dofs, total_dofs))
    f = np.zeros((total_dofs, 1))

    idx = elem.calculate_indices()
    K[np.ix_(idx, idx)] += elem.Ke
    f[idx] += fuerza.reshape(-1, 1)

    restrain_map = np.concatenate([n.restrain for n in nodos])
    libres = np.where(restrain_map == 'f')[0]

    Kff = K[np.ix_(libres, libres)]
    ff = f[libres]

    uf = np.linalg.solve(Kff, ff)
    u = np.zeros((total_dofs, 1))
    u[libres] = uf
    return u, f, K

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
