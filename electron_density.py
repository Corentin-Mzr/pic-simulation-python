import numpy as np


def compute_electron_density(x_vec: np.ndarray, nx: int, width: float, n0: float) -> np.ndarray:
    """
    :param x_vec: Position of particles as vector
    :param nx: Number of mesh cells
    :param width: Size of box
    :param n0: Average electron number density
    :return: Electron number density on the mesh, normalized
    """
    m = len(x_vec)
    dx = width / nx
    i = (x_vec // dx).astype(int)
    i1 = i + 1
    weight_i = (i1 * dx - x_vec) / dx
    weight_i1 = (x_vec - i * dx) / dx
    i1 %= nx

    # Compute density vector
    n_vec = np.bincount(i[:, 0], weights=weight_i[:, 0], minlength=nx)
    n_vec += np.bincount(i1[:, 0], weights=weight_i1[:, 0], minlength=nx)
    n_vec *= n0 * width / m / dx

    return n_vec
