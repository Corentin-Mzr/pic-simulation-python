import numpy as np


def compute_potential(n_vec: np.ndarray, dx: float, n0: float) -> np.ndarray:
    """
    :param n_vec: Density vector
    :param dx: Distance between mesh points
    :param n0: Average electron number density
    :return: Potential vector phi_vec
    """
    # Create matrix for computation (numerical scheme for laplacian)
    n = len(n_vec)
    up_diag = np.diag(np.diag(np.eye(n - 1)), k=1)
    mat = -2 * np.eye(n) + up_diag + up_diag.T
    mat[0, -1] = 1
    mat[-1, 0] = 1
    mat = 1/dx**2 * mat

    # Compute potential vector
    phi_vec = np.linalg.inv(mat) @ (n_vec - n0).reshape((n, 1))
    return phi_vec
