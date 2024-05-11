import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from parameters import *
from electron_density import compute_electron_density
from potential import compute_potential
from electric_field import compute_electric_field
from acceleration import compute_acceleration

matplotlib.use('TkAgg')


def main():
    v_th = 1.0
    v_b = 0.1
    a = 0.01

    # Define initial position
    x_vec = width * np.random.rand(m, 1)

    # Define initial velocity
    # Two beams going in opposite directions
    v_vec = np.random.normal(loc=v_th, scale=v_b, size=(m, 1))
    v_vec[m//2:] *= -1.0

    # Add perturbations for faster merging
    v_vec *= (1 + a * np.sin(2 * np.pi * x_vec / width))

    # Define initial acceleration
    acc_vec = np.zeros((m, 1))
    t = 0

    fig, ax = plt.subplots(nrows=2, ncols=1)

    # Main loop
    for i in range(nt):
        ##### Leap from scheme #####

        # Compute new velocity
        v_vec += acc_vec * dt / 2.0

        # Compute new position + boundary conditions
        x_vec += v_vec * dt
        x_vec %= width

        ## Update acceleration ##

        # Compute density
        n_vec = compute_electron_density(x_vec, nx, width, n0)

        # Compute potential
        phi_vec = compute_potential(n_vec, dx, n0)

        # Compute electric field
        e_vec = compute_electric_field(phi_vec, dx)

        # Compute acceleration
        acc_vec = compute_acceleration(x_vec, nx, width, e_vec)

        #########################

        # Update velocity again
        v_vec += acc_vec * dt / 2.0

        # Update time
        t += dt

        ############################

        # Plot #
        ax[0].cla()
        ax[0].scatter(x_vec[:m // 2], v_vec[:m // 2], s=0.4, c='r')
        ax[0].scatter(x_vec[m // 2:], v_vec[m // 2:], s=0.4, c='b')
        ax[0].axis((0, width, -2 * v_th, 2 * v_th))
        ax[0].set_title(f"{t:.2f}s")

        ax[1].cla()
        ax[1].plot([k * dx * width for k in range(nx)], n_vec, 'g')
        ax[1].axis((0, width, 0, 5.0))
        plt.pause(1/nt)
    plt.show()


if __name__ == "__main__":
    main()
