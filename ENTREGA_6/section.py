import numpy as np

class Section:
    def __init__(self, thickness, E, nu):
        self.thickness = thickness
        self.E = E
        self.nu = nu
        self.D = self._compute_D_orthotropic()

    def _compute_D_orthotropic(self):
        E = self.E
        nu = self.nu
        return (E / (1 - nu**2)) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ])
    