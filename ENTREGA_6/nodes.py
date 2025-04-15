import numpy as np
import matplotlib.pyplot as plt



class Node:
    def __init__(self, id, x, y, dofs=None, restrain=[0, 0]):
        self.id = id
        self.x = x
        self.y = y
        if dofs is None:
            self.dofs = np.array([id * 2, id * 2 + 1])
        else:
            self.dofs = np.array(dofs)
        self.restrain = np.array(restrain)


    def plot_nodes(nodes, show_ids=True):
        """
        Plotea una lista de nodos en 2D.

        Parámetros:
        - nodes: lista de objetos Node
        - show_ids: si es True, muestra los IDs junto a los nodos
        """
        xs = [node.x for node in nodes]
        ys = [node.y for node in nodes]

        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, c='blue', s=10, label='Nodos')

        if show_ids:
            for node in nodes:
                plt.text(node.x + 0.5, node.y + 0.5, str(node.id), fontsize=5, color='red')

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Distribución de Nodos")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

    def plot_nodes_por_grupo(grupo_nodos_dict, show_ids=False):
        """
        Plotea nodos por grupo con colores diferentes.

        Parámetros:
        - grupo_nodos_dict: dict con nombre del grupo como clave y lista de nodos como valor
        - show_ids: muestra IDs si True
        """
        plt.figure(figsize=(7, 7))
        colors = plt.cm.get_cmap("tab10", len(grupo_nodos_dict))

        for i, (grupo, nodos) in enumerate(grupo_nodos_dict.items()):
            xs = [n.x for n in nodos]
            ys = [n.y for n in nodos]
            plt.scatter(xs, ys, s=2, color=colors(i), label=grupo)

            if show_ids:
                for node in nodos:
                    plt.text(node.x + 0.5, node.y + 0.5, str(node.id), fontsize=6)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Nodos por grupo físico")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()



    

   