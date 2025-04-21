import numpy as np

class Solve:
    def __init__(self, nodes, elements):
        self.nodes = nodes

           


        self.elements = elements
        self.ndof = len(nodes) * 2  # 2 DOFs por nodo
        self.K_global = np.zeros((self.ndof, self.ndof))
        self.f_global = np.zeros((self.ndof, 1))
        self.u_global = np.zeros((self.ndof, 1))

    def assemble(self):
        for elem in self.elements:
            ke = elem.Ke
            idx = elem.calculate_indices()
            for i in range(6):
                for j in range(6):
                    self.K_global[idx[i], idx[j]] += ke[i, j]

    def apply_force(self, dof_index, value):
        self.f_global[dof_index] += value

    def apply_forces_vector(self, force_vector):
        self.f_global += force_vector.reshape(-1, 1)

    def apply_boundary_conditions(self):
        restrain_map = np.concatenate([n.restrain for n in self.nodes])
        self.free_dofs = np.where(restrain_map == 0)[0]
        self.fixed_dofs = np.where(restrain_map == 1)[0]

        print("N° de DOFs libres:", len(self.free_dofs))
        print("N° de DOFs fijos:", len(self.fixed_dofs))


        # Aplicar condiciones
        for dof in self.fixed_dofs:
            self.K_global[dof, :] = 0
            self.K_global[:, dof] = 0
            self.K_global[dof, dof] = 1
            self.f_global[dof] = 0

    def check_zero_rows(self):
        zero_rows = np.where(~self.K_global.any(axis=1))[0]
        if len(zero_rows) > 0:
            print(f"❌ Fila(s) completamente nulas en K_global: {zero_rows}")
        else:
            print("✅ No hay filas nulas en K_global")

        return zero_rows


    def solve(self):
        self.assemble()

        self.apply_boundary_conditions()
        self.u_global = np.linalg.solve(self.K_global, self.f_global)
        return self.u_global

    def get_displacement_at_node(self, node_id):
        node = self.nodes[node_id - 1]  # Asume que el ID parte en 1
        ux = self.u_global[node.dofs[0], 0]
        uy = self.u_global[node.dofs[1], 0]
        return ux, uy

    def print_summary(self):
        print("Desplazamientos globales:")
        for i, node in enumerate(self.nodes):
            ux, uy = self.get_displacement_at_node(node.id)
            print(f"Nodo {node.id}: ux = {ux:.4e} mm, uy = {uy:.4e} mm")
