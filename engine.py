import numpy as np
import matplotlib.pyplot as plt
import pandas
import json
from pressure_corrector import successiveOverRelaxation

Lx = 10
Ly = 10

nx = 10
ny = 10

dx = Lx / nx
dy = Ly / ny

x_velocity_field = np.ones((nx + 1, ny + 2))
y_velocity_field = np.ones((nx + 2, ny + 1))

x_star_field = np.ones((nx + 1, ny + 2))
y_star_field = np.ones((nx + 2, ny + 1))

x_flux = np.ones((nx + 1, ny + 2))
y_flux = np.ones(((nx + 1, ny + 1)))

pressure = np.ones((nx + 2, ny + 2)) # Cell centered

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

prhs = np.zeros_like(pressure)

mu = 1e-5
rho = 1.0

dt = 0.01 # secs

def computeFluxes(
        x_velocity_field: np.array, 
        y_velocity_field: np.array, 
        x_flux: np.array, 
        y_flux: np.array
    ) -> None:

    print(x_velocity_field[0, 9])
    print(y_velocity_field.shape)
    # x-momentum
    for i in range(1, nx):
        for j in range(1, ny):
            x_flux[i, j] = dy * (((x_velocity_field[i + 1, j] + x_velocity_field[i, j]) / 2)**2 - mu * ((x_velocity_field[i + 1, j] - x_velocity_field[i, j]) / dx))        # Eastern Flux

            x_flux[i - 1, j] = -dy * (((x_velocity_field[i, j] + x_velocity_field[i - 1, j]) / 2)**2 - mu * ((x_velocity_field[i, j] - x_velocity_field[i - 1, j]) / dx))   # Western Flux

            y_flux[i, j] = dx * (((x_velocity_field[i, j + 1] + x_velocity_field[i, j]) / 2) * ((y_velocity_field[i + 1, j] + y_velocity_field[i, j]) / 2) - \
                                  mu * ((x_velocity_field[i, j + 1] - x_velocity_field[i, j]) / dy))                                                                        # Northern Flux

            y_flux[i, j - 1] = -dx * (((x_velocity_field[i, j] + x_velocity_field[i, j - 1]) / 2) * ((y_velocity_field[i + 1, j - 1] + y_velocity_field[i, j - 1]) / 2) - \
                                      mu * ((x_velocity_field[i, j] - x_velocity_field[i, j - 1]) / dy))                                                                    # Southern Flux

    # TODO: Add y-momentum

def computeStarredVelocities(
        x_flux: np.array,
        y_flux: np.array,
        x_velocity_field: np.array,
        x_star_field: np.array,
        y_star_field: np.array
    ) -> None:

    for i in range(1, nx):
        for j in range(1, ny):
            x_star_field[i, j] = x_velocity_field[i, j] + (dt / (dx * dy)) * (x_flux[i, j] + x_flux[i - 1, j] + y_flux[i, j] + y_flux[i, j - 1])

    # TODO: Add y_star_field once y-momentum is added

def correctVelocities(
        x_velocity_field: np.array,
        y_velocity_field: np.array,
        x_star_field: np.array,
        y_star_field: np.array,
        pressure: np.array
    ) -> None:
    
    for i in range(1, nx + 1):
        for j in range(1, nx + 1):
            prhs[i, j] = (rho / dt) * ((x_star_field[i, j] - x_star_field[i - 1, j]) / dx + (y_star_field[i, j] - y_star_field[i, j - 1]) / dy)

    beta = 1.2

    pressure, error = successiveOverRelaxation(beta,  nx, ny, pressure, prhs, coeffs)

    # Correct the u velocity
    for i in range(1, nx):
        for j in range(1, ny + 1):
            x_velocity_field[i, j] = x_star_field[i, j] - dt / dx * (pressure[i + 1, j] - pressure[i, j]) / rho
            
    for i in range(1, nx + 1):
        for j in range(1, ny):
            y_velocity_field[i, j] = y_star_field[i, j] - dt / dy * (pressure[i, j + 1] - pressure[i, j]) / rho

    plt.contour(y_velocity_field)
    plt.show()


computeFluxes(x_velocity_field, y_velocity_field, x_flux, y_flux)
computeStarredVelocities(x_flux, y_flux, x_velocity_field, x_star_field, y_star_field)
correctVelocities(x_velocity_field, y_velocity_field, x_star_field, y_star_field, pressure)




