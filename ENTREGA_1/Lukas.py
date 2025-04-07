import numpy as np
import matplotlib.pyplot as plt

def cst(xy, properties, point, ue=None, Strain = False, Stress = False, E_matrix = None):

    if Strain and Stress:
        raise ValueError("Strain and Stress cannot be both True at the same time.")
    
    E_mod = properties['E']
    rho = properties['rho']
    poisson = properties['poisson']
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

    chi = mobilus_coordinates(xy, point)
    area = element_area(xy)

    B = np.array([
        [chi[0], 0,       chi[1], 0,       chi[2], 0      ],
        [0,       chi[0], 0,       chi[1], 0,       chi[2]],
        [chi[0], chi[0],  chi[1], chi[1],  chi[2], chi[2]]
    ])

    if Strain:

        E = (E/(1 - poisson**2)) * np.array([[1, poisson, 0],
                                             [poisson, 1, 0],
                                             [0, 0, (1 - poisson)/2]])
        
    elif Stress:
        E = E_matrix

    else:
        E = E_matrix
        


    ke = (B.T @ E @ B) * area * t

    print("Stiffness Matrix (ke):")
    print(ke)

    









# Example usage
if __name__ == "__main__":
    properties = {
        'E': 210e9,
        'rho': 7850,
        'poisson': 0.3
    }

    xy = [[0, 0], [3, 1], [2, 2]]
    point = [0.5, 0.5]  # Point where the mobilus coordinates are calculated
    ue = [1, 0]  # Optional: displacement

    E_matrix = np.array([[32, 8, 0],
                         [0, 32, 0],
                         [0, 0, 16]])  # Optional: E matrix for stress calculation

    cst(xy, properties, point, ue, E_matrix=E_matrix)


