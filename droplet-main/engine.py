import numpy as np
import matplotlib.pyplot as plt
import pandas
import json
from pressure_corrector import successiveOverRelaxation


def computeFluxes(
        T,
        rho: float,
        mu: float,
        nx: int,
        ny: int,
        dt: float,
        gx: float,
        gy: float,
        dx: float,
        dy: float,
        x_velocity_field: np.array, 
        y_velocity_field: np.array, 
        x_flux_x: np.array, 
        y_flux_x: np.array,
        x_flux_y: np.array,
        y_flux_y: np.array,
        x_star_field: np.array,
        y_star_field: np.array
    ) -> None:

    # x-momentum contribution to velocity
    for i in range(1, nx):
        for j in range(1, ny + 1):
            east_flux = 0.25 * (x_velocity_field[i + 1, j] + x_velocity_field[i, j])**2 - mu * (x_velocity_field[i + 1, j] - x_velocity_field[i, j]) / dx
            north_flux = 0.25 * (y_velocity_field[i + 1, j] + y_velocity_field[i, j]) * (x_velocity_field[i, j + 1] + x_velocity_field[i, j]) - mu * (x_velocity_field[i, j + 1] - x_velocity_field[i, j]) / dy
            west_flux = 0.25 * (x_velocity_field[i, j] + x_velocity_field[i - 1, j])**2 - mu * (x_velocity_field[i, j] - x_velocity_field[i - 1, j]) / dx
            south_flux = 0.25 * (y_velocity_field[i, j - 1] + y_velocity_field[i + 1, j - 1]) * (x_velocity_field[i, j] + x_velocity_field[i, j - 1]) - mu * (x_velocity_field[i, j] - x_velocity_field[i, j - 1]) / dy
            
            x_star_field[i, j] = x_velocity_field[i, j] - (dt / (dx * dy)) * (dy * (east_flux - west_flux) + dx * (north_flux - south_flux)) + dt * gx 
            # Note gx is zero so no boussenique here

    # y-momentum contribution to velocity   
    for i in range(1, nx + 1):
        for j in range(1, ny):

            # Compute cell fluxes
            east_flux = 0.25 * (x_velocity_field[i, j + 1] + x_velocity_field[i, j]) * (y_velocity_field[i + 1, j] + y_velocity_field[i, j]) - mu * (y_velocity_field[i + 1, j] - y_velocity_field[i, j]) / dx
            north_flux = 0.25 * (y_velocity_field[i, j + 1] + y_velocity_field[i, j])**2 - mu * (y_velocity_field[i, j + 1] - y_velocity_field[i, j]) / dy
            west_flux = 0.25 * (x_velocity_field[i - 1, j + 1] + x_velocity_field[i - 1, j]) * (y_velocity_field[i, j] + y_velocity_field[i - 1, j]) - mu * (y_velocity_field[i, j] - y_velocity_field[i - 1, j]) / dx
            south_flux = 0.25 * (y_velocity_field[i, j] + y_velocity_field[i, j - 1])**2 - mu * (y_velocity_field[i, j] - y_velocity_field[i, j - 1]) / dy

            # Compute estimation (v*) of velocity
            y_star_field[i, j] = y_velocity_field[i, j] - (dt / (dx * dy)) * (dy * (east_flux - west_flux) + dx * (north_flux - south_flux)) + dt * gy

            # Compute body force from natural convection
            #y_star_field[i, j] =- dt * 0.002 * (((T[i - 1, j] + T[i - 1, j - 1]) / 2) - 300) * gy

def correctVelocities(
        dt: float,
        coeffs: dict,
        rho: float,
        mu: float,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        prhs: np.array,
        x_velocity_field: np.array,
        y_velocity_field: np.array,
        x_star_field: np.array,
        y_star_field: np.array,
        pressure: np.array
    ) -> None:
    
    prhs[1:nx + 1, 1:ny + 1] = (rho / dt) * ((x_star_field[1:nx + 1, 1:ny + 1] - x_star_field[:nx, 1:ny + 1]) / dx + (y_star_field[1:nx + 1, 1:ny + 1] - y_star_field[1:nx + 1, :ny]) / dy)

    beta = 1.2

    pressure, error = successiveOverRelaxation(beta,  nx, ny, pressure, prhs, coeffs)

    x_velocity_field[1:nx, 1:ny + 1] = x_star_field[1:nx, 1:ny + 1] - dt / dx * (pressure[2:nx + 1, 1:ny + 1] - pressure[1:nx, 1:ny + 1]) / rho       
    y_velocity_field[1:nx + 1, 1:ny] = y_star_field[1:nx + 1, 1:ny] - dt / dy * (pressure[1:nx + 1, 2:ny + 1] - pressure[1:nx + 1, 1:ny]) / rho

def computeTemperatureField(
        dt, 
        temp_bc: dict,
        nx, ny, dx, dy,
        x_velocity_field: np.array,
        y_velocity_field: np.array,
        x_flux: np.array,
        y_flux: np.array,
        alpha: float,
        T: np.array
    ) -> None:

    T_wall_t = 300
    T_wall_b = 1000

    x_flux[1:-1, :] = (((T[1:, :] + T[:-1, :]) / 2) * x_velocity_field[1:nx, 1:nx + 1]) - alpha * (T[1:, :] - T[:-1, :]) / dx
    y_flux[:, 1:-1] = (((T[:, 1:] + T[:, :-1]) / 2) * y_velocity_field[1:nx + 1, 1:ny]) - alpha * (T[:, 1:] - T[:, :-1]) / dy 

    x_flux[0, :] = ((T[0, :])) * x_velocity_field[0, 1:ny + 1]
    x_flux[-1, :] = ((T[-1, :])) * x_velocity_field[-1, 1:ny + 1]

    y_flux[:, 0] = ((T_wall_b + T[:, 0]) / 2) * y_velocity_field[1:nx + 1, 0] - alpha * (T[:, 0] - T_wall_b) / (dy / 2.0)
    y_flux[:, -1] = ((T_wall_t + T[:, -1]) / 2) * y_velocity_field[1:nx + 1, -1] - alpha * (T_wall_t - T[:, -1]) / (dy / 2.0)

    T[:, :] = T[:, :] - (dt / (dx * dy)) * (dy * (x_flux[1:, :] - x_flux[:-1, :]) + dx * (y_flux[:, 1:] - y_flux[:, :-1]))

def applyBoundaryConditions(
        bc: dict,
        x_velocity_field: np.array,
        y_velocity_field: np.array,
        x_flux_x,
        y_flux_x,
        x_flux_y,
        y_flux_y,
        bc_mask: np.array = None,
    ) -> None:

    x_velocity_field[:, 0] = 2 * bc["usouth"] - x_velocity_field[:, 1]
    x_velocity_field[:, -1] = 2 * bc["unorth"] - x_velocity_field[:, -2]
    y_velocity_field[0, :] = 2 * bc["vwest"] - y_velocity_field[1, :]
    y_velocity_field[-1, :] = 2 * bc["veast"] - y_velocity_field[-2, :]

    y_velocity_field[:, -1] = bc["vnorth"]
    y_velocity_field[:, 0] = bc["vsouth"]
    x_velocity_field[0, :] = bc["uwest"]
    x_velocity_field[-1, :] = bc["ueast"]

    base_mask = bc_mask[:, :]
    right_shifted = np.roll(bc_mask, 1, 0)
    up_shifted = np.roll(bc_mask, 1, 1)

    # Mask out 'dead cell fluxes' to create obstacles if specified, ignore if not specified or mask is all False
    if bc_mask is not None:
        x_velocity_field[np.pad(base_mask, ((1, 0), (1, 1)), mode = 'constant', constant_values = False)] = 0
        x_velocity_field[np.pad(up_shifted, ((1, 0), (1, 1)), mode = 'constant', constant_values = False)] = 0

        y_velocity_field[np.pad(base_mask, ((1, 1), (1, 0)), mode = 'constant', constant_values = False)] = 0
        y_velocity_field[np.pad(right_shifted, ((1, 1), (1, 0)), mode = 'constant', constant_values = False)] = 0




if __name__ == "__main__":

    Lx = 1
    Ly = 1

    nx = 10
    ny = 20

    dx = Lx / nx
    dy = Ly / ny

    usouth = -1
    unorth = 1
    vwest = 0
    veast = 0
    vsouth = 0
    vnorth = 0
    uwest = 0
    ueast = 0

    x_velocity_field = np.zeros((nx + 1, ny + 2))
    y_velocity_field = np.zeros((nx + 2, ny + 1))

    x_star_field = np.zeros((nx + 1, ny + 2))
    y_star_field = np.zeros((nx + 2, ny + 1))

    x_flux_x = np.zeros((nx + 1, ny + 2))
    y_flux_x = np.zeros(((nx + 2, ny + 1)))
    x_flux_y = np.zeros((nx + 1, ny + 2))
    y_flux_y = np.zeros(((nx + 2, ny + 1)))

    pressure = np.zeros((nx + 2, ny + 2)) # Cell centered

    A_E = np.ones_like(pressure) * (1/dx**2)
    A_W = np.ones_like(pressure) * (1/dx**2)
    A_N = np.ones_like(pressure) * (1/dy**2)
    A_S = np.ones_like(pressure) * (1/dy**2)

    A_W[1, :] = 0.0
    A_E[-2, :] = 0.0
    A_S[:, 1] = 0.0
    A_N[:, -2] = 0.0

    gx = 0.0
    gy = -9.81

    uu = np.zeros((nx + 1, ny + 1))
    vv = np.zeros((nx + 1, ny + 1))
    uold = uu.copy()
    vold = vv.copy()

    xnodes = np.linspace(dx / 2, Lx - dx / 2, nx)
    ynodes = np.linspace(dy / 2, Ly - dy / 2, ny)

    #X, Y = np.meshgrid(x, y, indexing = "ij")
    cell_vel = np.zeros((nx, ny))

    A_P = A_E + A_W + A_S + A_N

    coeffs = {"A_E": A_E, "A_W": A_W, "A_S": A_S, "A_N": A_N, "A_P": A_P}

    prhs = np.zeros_like(pressure)

    mu = 0.01 / 1
    rho = 1.0

    dt = 0.01 # secs

    fig, ax = plt.subplots(1, 1)

    xx, yy = np.meshgrid(xnodes, ynodes, indexing = "ij")
    for i in range(10000):
        applyBoundaryConditions(x_velocity_field, y_velocity_field)
        computeFluxes(x_velocity_field, y_velocity_field, x_flux_x, y_flux_x, x_flux_y, y_flux_y, x_star_field, y_star_field)
        computeStarredVelocities(x_flux_x, y_flux_x, x_flux_y, y_flux_y, x_velocity_field, x_star_field, y_star_field)
        correctVelocities(x_velocity_field, y_velocity_field, x_star_field, y_star_field, pressure)

        if (i % 50) == 0:
            uu = 0.5 * (x_velocity_field[1:, 1:-1] + x_velocity_field[:-1, 1:-1])
            vv = 0.5 * (y_velocity_field[1:-1, 1:] + y_velocity_field[1:-1, :-1])
            print(i)
            
            ax.contourf(xx, yy, np.sqrt(uu * uu + vv * vv), vmin = 0.0, vmax = 1.0)
            ax.quiver(xx, yy, uu, vv)
            plt.pause(0.001)

        x_star_field[:, :] = 0
        y_star_field[:, :] = 0
    plt.show()
