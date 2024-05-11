import numpy as np


def compute_electric_field(phi_vec: np.ndarray, dx: float) -> np.ndarray:
    """
    :param phi_vec: Potential vector
    :param dx: Distance between mesh points
    :return: Electric field vector e_vec
    """
    # Create matrix for computation (numerical scheme for gradient)
    n = len(phi_vec)
    up_diag = np.diag(np.diag(np.eye(n - 1)), k=1)
    mat = np.zeros((n, n)) + up_diag - up_diag.T
    mat[-1, 0] = 1
    mat[0, -1] = -1
    mat = -1/(2 * dx) * mat

    # Compute electric field vector
    e_vec = mat @ phi_vec

    return e_vec
