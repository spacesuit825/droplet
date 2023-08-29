import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK

def successiveOverRelaxation(
        beta: float,
        nx: int, 
        ny: int, 
        pressure: float, 
        prhs: float,
        coeffs: dict
    ) -> tuple([np.array, float]):

    Ae = coeffs["A_E"]
    Aw = coeffs["A_W"]
    An = coeffs["A_N"]
    As = coeffs["A_S"]
    Ap = coeffs["A_P"]

    iteration = 0
    max_iterations = 1000
    error = 1e5
    tol = 1e-5
    
    while error > tol and iteration < max_iterations:
        p_previous = pressure.copy()
        
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                pressure[i, j] = (1 - beta) * pressure[i, j] + (beta / Ap[i, j]) * (Ae[i, j] * pressure[i + 1, j] + Aw[i, j] * pressure[i - 1, j] + \
                                                                                    An[i, j] * pressure[i, j + 1] + As[i, j] * pressure[i, j - 1] - \
                                                                                    prhs[i, j])

        error = np.linalg.norm(pressure - p_previous)

        iteration += 1
    
    if iteration == max_iterations:
        print(f"Successive Over-Relaxation did not converge! Last iteration error was {error}...")

    return pressure, error


if __name__ == "__main__":
    
    velocity_import = np.load("C:/Users/lachl/Documents/python/droplet/data/lid_driven_cavity_nondivfree_velocities.npz")

    ut = velocity_import["ut"]
    vt = velocity_import["vt"]
    pressure = velocity_import["p"]
    u = velocity_import["u"]
    v = velocity_import["v"]

    nx = ut.shape[0] - 1
    ny = ut.shape[1] - 2

    print(nx, ny)
    print(u.shape)
    print(v.shape)

    rho, Lx, Ly = 1.0, 1.0, 1.0

    dx = Lx / nx
    dy = Ly / ny

    dt = 0.01

    prhs = np.zeros_like(pressure)

    for i in range(1, nx + 1):
        for j in range(1, nx + 1):
            prhs[i, j] = (rho / dt) * ((ut[i, j] - ut[i - 1, j]) / dx + (vt[i, j] - vt[i, j - 1]) / dy)

    A_E = np.ones_like(pressure) * (1/dx**2)
    A_W = np.ones_like(pressure) * (1/dx**2)
    A_N = np.ones_like(pressure) * (1/dy**2)
    A_S = np.ones_like(pressure) * (1/dy**2)

    A_W[1, :] = 0.0
    A_E[-2, :] = 0.0
    A_S[:, 1] = 0.0
    A_N[:, -2] = 0.0

    A_P = A_E + A_W + A_S + A_N

    coeffs = {"A_E": A_E, "A_W": A_W, "A_S": A_S, "A_N": A_N, "A_P": A_P}

    beta = 1.2

    pressure, error = successiveOverRelaxation(beta,  nx, ny, pressure, prhs, coeffs)

    # Correct the u velocity
    for i in range(1, nx):
        for j in range(1, ny + 1):
            u[i, j] = ut[i, j] - dt / dx * (pressure[i + 1, j] - pressure[i, j]) / rho
            
    for i in range(1, nx + 1):
        for j in range(1, ny):
            v[i, j] = vt[i, j] - dt / dy * (pressure[i, j + 1] - pressure[i, j]) / rho

    divuu = (u[1:, 1:-1] - u[:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy
    print("divuu: ", np.linalg.norm(divuu))
    print(error)

    # # print(x_velocity_field)
    # for i in range(1, nx):
    #     for j in range(1, ny + 1):
    #         east_flux = 0.25 * (x_velocity_field[i + 1, j] + x_velocity_field[i, j])**2 - mu * (x_velocity_field[i + 1, j] - x_velocity_field[i, j]) / dx
    #         north_flux = 0.25 * (y_velocity_field[i + 1, j] + y_velocity_field[i, j]) * (x_velocity_field[i, j + 1] + x_velocity_field[i, j]) - mu * (x_velocity_field[i, j + 1] - x_velocity_field[i, j]) / dy
    #         west_flux = 0.25 * (x_velocity_field[i, j] + x_velocity_field[i - 1, j])**2 - mu * (x_velocity_field[i, j] - x_velocity_field[i - 1, j]) / dx
    #         south_flux = 0.25 * (y_velocity_field[i, j - 1] + y_velocity_field[i + 1, j - 1]) * (x_velocity_field[i, j] + x_velocity_field[i, j - 1]) - mu * (x_velocity_field[i, j] - x_velocity_field[i, j - 1]) / dy
    #         # bed_e[i, j] = east_flux
    #         # bed_w[i, j] = north_flux
    #         # check[i, j] = 0.25 * (x_velocity_field[i + 1, j] + x_velocity_field[i, j])**2
    #         check2[i, j] = - mu * (x_velocity_field[i + 1, j] - x_velocity_field[i, j]) / dx
    #         x_star_field[i, j] = x_velocity_field[i, j] - (dt / (dx * dy)) * (dy * (east_flux - west_flux) + dx * (north_flux - south_flux)) + dt * gx

    # bed_e = np.zeros((nx + 1, ny))
    # bed_w = np.zeros((nx + 1, ny))
    # for i in range(1, nx + 1):
    #     for j in range(1, ny):
    #         east_flux = 0.25 * (x_velocity_field[i, j + 1] + x_velocity_field[i, j]) * (y_velocity_field[i + 1, j] + y_velocity_field[i, j]) - mu * (y_velocity_field[i + 1, j] - y_velocity_field[i, j]) / dx
    #         north_flux = 0.25 * (y_velocity_field[i, j + 1] + y_velocity_field[i, j])**2 - mu * (y_velocity_field[i, j + 1] - y_velocity_field[i, j]) / dy
    #         west_flux = 0.25 * (x_velocity_field[i - 1, j + 1] + x_velocity_field[i - 1, j]) * (y_velocity_field[i, j] + y_velocity_field[i - 1, j]) - mu * (y_velocity_field[i, j] - y_velocity_field[i - 1, j]) / dx
    #         south_flux = 0.25 * (y_velocity_field[i, j] + y_velocity_field[i, j - 1])**2 - mu * (y_velocity_field[i, j] - y_velocity_field[i, j - 1]) / dy
    #         bed_e[i, j] = east_flux
    #         bed_w[i, j] = north_flux
    #         check[i, j] = 0.25 * (x_velocity_field[i, j + 1] + x_velocity_field[i, j]) * (y_velocity_field[i + 1, j] + y_velocity_field[i, j]) - mu * (y_velocity_field[i + 1, j] - y_velocity_field[i, j]) / dx

    #         y_star_field[i, j] = y_velocity_field[i, j] - (dt / (dx * dy)) * (dy * (east_flux - west_flux) + dx * (north_flux - south_flux)) + dt * gy