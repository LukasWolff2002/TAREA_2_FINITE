import numpy as np

class Node:
    def __init__(self, id, x, y, dofs, restrain=[0, 0]):
        self.id = id
        self.x = x
        self.y = y
        self.dofs = np.array(dofs)
        self.restrain = np.array(restrain)