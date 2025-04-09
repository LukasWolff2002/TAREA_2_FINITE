import numpy as np
#ACA DEBE IR LA MATRIZ GLOBAL
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