import numpy as np


def compute_acceleration(x_vec: np.ndarray, nx: int, width: float, e_vec: np.ndarray) -> np.ndarray:
    """
    :param x_vec: Position of particles as vector
    :param nx: Number of mesh cells
    :param width: Size of box
    :param e_vec: Electric field vector
    :return: Acceleration vector
    """
    # Compute weights
    m = len(x_vec)
    dx = width / nx
    i = (x_vec//dx).astype(int)
    i1 = i + 1
    weight_i = (i1 * dx - x_vec) / dx
    weight_i1 = (x_vec - i * dx) / dx
    i1 %= nx

    # Compute electric field for each particle
    elec_vec = weight_i * e_vec[i.reshape((1, -1))] + weight_i1 * e_vec[i1.reshape((1, -1))]

    # a(ri) = -E(ri)
    acc_vec = - elec_vec.reshape((m, 1))

    return acc_vec
