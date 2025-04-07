import numpy as np
import matplotlib.pyplot as plt

def cst(xy, properties, ue=None):
    E_mod = properties['E']
    rho = properties['rho']
    t = 1  # Thickness of the element (assumed constant for simplicity)

    def mobilus_coordinates(xy, point):
        A = np.array([point[0], point[1], 1])
        B = np.array([[xy[0][0], xy[1][0], xy[2][0]],
                      [xy[0][1], xy[1][1], xy[2][1]],
                      [1, 1, 1]])
        x = np.linalg.solve(B, A)
        return x

    def element_area(xy):
        A = np.array([[1, xy[0][0], xy[0][1]],
                      [1, xy[1][0], xy[1][1]],
                      [1, xy[2][0], xy[2][1]]])
        return 0.5 * np.abs(np.linalg.det(A))

    chi = mobilus_coordinates(xy, [0.5, 0.5]) #coordenadas triangulares para ese punto (0,5 ; 0,5)
    area = element_area(xy)

    B = np.array([
        [chi[0], 0,       chi[1], 0,       chi[2], 0      ],
        [0,       chi[0], 0,       chi[1], 0,       chi[2]],
        [chi[0], chi[0],  chi[1], chi[1],  chi[2], chi[2]]
    ])

    E = np.array([
        [E_mod, 0,     0    ],
        [0,     E_mod, 0    ],
        [0,     0,     E_mod]
    ])

    ke = (B.T @ E @ B) * area * t

    if ue is not None:
        ue = np.array(ue).flatten()
        ue = np.tile(ue, 3)  # Expand to 6 components if only 2 are given
        fe = (area * t / 3) * ue.reshape((6, 1))

        epsilon = B @ ue
  
        sigma = E @ epsilon
    else:
        fe = None

    return ke, fe






def plot_element(xy, title="Triangular Element"):
    xy = np.array(xy + [xy[0]])  # Cierra el triángulo
    plt.figure(figsize=(5,5))
    plt.plot(xy[:,0], xy[:,1], 'bo-')
    for i, (x, y) in enumerate(xy[:-1]):
        plt.text(x, y, f'N{i+1}', fontsize=12, ha='right')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    properties = {
        'E': 210e9,
        'rho': 7850
    }

    xy = [[0, 0], [1, 0], [1, 1]]
    ue = [1, 0]  # Optional: displacement

    ke, fe = cst(xy, properties, ue)

    print("Stiffness matrix ke:\n", ke)
    if fe is not None:
        print("Equivalent nodal force vector fe:\n", fe)

    plot_element(xy)
