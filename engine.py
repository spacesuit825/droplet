import numpy as np
import matplotlib.pyplot as plt
import pandas
import json
from pressure_corrector import successiveOverRelaxation

Lx = 1
Ly = 1

nx = 20
ny = 20

dx = Lx / nx
dy = Ly / ny

usouth = 0
unorth = 1
vwest = 0
veast = 1
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

def computeFluxes(
        x_velocity_field: np.array, 
        y_velocity_field: np.array, 
        x_flux_x: np.array, 
        y_flux_x: np.array,
        x_flux_y: np.array,
        y_flux_y: np.array,
        x_star_field: np.array,
        y_star_field: np.array
    ) -> None:
     
    x_flux_x[1:nx, 1:ny + 1] = 0.25 * (x_velocity_field[2:nx + 1, 1:ny + 1] + x_velocity_field[1:nx, 1:ny + 1])**2 - mu * (x_velocity_field[2:nx + 1, 1:ny + 1] - x_velocity_field[1:nx, 1:ny + 1]) / dx
    
    y_flux_x[1:nx, 1:ny + 1] = 0.25 * (y_velocity_field[2:nx + 1, 1:ny + 1] + y_velocity_field[1:nx, 1:ny + 1]) * (x_velocity_field[1:nx, 2:ny + 2] + x_velocity_field[1:nx, 1:ny + 1]) - mu * (x_velocity_field[1:nx, 2:ny + 2] - x_velocity_field[1:nx, 1:ny + 1]) / dy
    
    x_flux_y[1:nx + 1, 1:ny] = 0.25 * (x_velocity_field[1:ny + 1, 2:nx + 1] + x_velocity_field[1:nx + 1, 1:ny]) * (y_velocity_field[2:nx + 2, 1:ny] + y_velocity_field[1:nx + 1, 1:ny]) - mu * (y_velocity_field[2:nx + 2, 1:ny] - y_velocity_field[1:nx +1, 1:ny]) / dx
    y_flux_y[1:nx + 1, 1:ny] = 0.25 * (y_velocity_field[1:ny + 1, 2:nx + 1] + y_velocity_field[1:nx + 1, 1:ny])**2 - mu * (y_velocity_field[1:ny + 1, 2:nx + 1] - y_velocity_field[1:nx + 1, 1:ny]) / dy

def computeStarredVelocities(
        x_flux_x: np.array,
        y_flux_x: np.array,
        x_flux_y: np.array,
        y_flux_y: np.array,
        x_velocity_field: np.array,
        x_star_field: np.array,
        y_star_field: np.array
    ) -> None:

    for i in range(1, nx):
        for j in range(1, ny + 1):
            x_star_field[i, j] = x_velocity_field[i, j] - (dt / (dx * dy)) * (dy * (x_flux_x[i, j] - x_flux_x[i - 1, j]) + dx * (y_flux_x[i, j] - y_flux_x[i, j - 1])) + dt * gx

    for i in range(1, nx + 1):
        for j in range(1, ny):
            y_star_field[i, j] = y_velocity_field[i, j] - (dt / (dx * dy)) * (dy * (x_flux_y[i, j] - x_flux_y[i - 1, j]) + dx * (y_flux_y[i, j] - y_flux_y[i, j - 1])) + dt * gy

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

def applyBoundaryConditions(
        x_velocity_field: np.array,
        y_velocity_field: np.array
    ) -> None:

    x_velocity_field[:, 0] = 2 * usouth - x_velocity_field[:, 1]
    x_velocity_field[:, -1] = 2 * unorth - x_velocity_field[:, -2]
    y_velocity_field[0, :] = 2 * vwest - y_velocity_field[1, :]
    y_velocity_field[-1, :] = 2 * veast - y_velocity_field[-2, :]

    y_velocity_field[:, -1] = vnorth
    y_velocity_field[:, 0] = vsouth
    x_velocity_field[0, :] = uwest
    x_velocity_field[-1, :] = ueast

xx, yy = np.meshgrid(xnodes, ynodes, indexing = "ij")
for i in range(1000):
    applyBoundaryConditions(x_velocity_field, y_velocity_field)
    computeFluxes(x_velocity_field, y_velocity_field, x_flux_x, y_flux_x, x_flux_y, y_flux_y, x_star_field, y_star_field)
    computeStarredVelocities(x_flux_x, y_flux_x, x_flux_y, y_flux_y, x_velocity_field, x_star_field, y_star_field)
    correctVelocities(x_velocity_field, y_velocity_field, x_star_field, y_star_field, pressure)
    
    if (i % 50) == 0:
        uu = 0.5 * (x_velocity_field[1:, 1:-1] + x_velocity_field[:-1, 1:-1])
        vv = 0.5 * (y_velocity_field[1:-1, 1:] + y_velocity_field[1:-1, :-1])
        print(i)
        
        print(np.linalg.norm(uu))
        ax.contourf(xx, yy, np.sqrt(uu * uu + vv * vv), vmin = 0.0, vmax = 1.0)
        ax.quiver(xx, yy, uu, vv)
        plt.pause(0.001)
plt.show()
    #computeCellCentredVelocities(x_velocity_field, y_velocity_field)




