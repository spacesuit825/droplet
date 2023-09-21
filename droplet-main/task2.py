import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK


Lx = 1.0 # m
Ly = 0.25 # m
alpha = 0.01 # Diffusivity coeff

mu = 1e-5 / 1
p_grad = 0.001

## GRID
nx = 20
ny = 20
dx = Lx / nx # m
dy = Ly / ny # m

## TIME
time = 0.0 # secs
dt = 1e-3 # secs
time_end = 5.0 # secs

plotting = 10

T_cool = 300.0 # Initial condition K
T_wall_t = 350 # Top Wall condition K
T_wall_l = 300.0 # Left Wall condition K

S = -2000

max_temp = []
time_t = np.linspace(0, )

x = np.linspace(dx/2, Lx - dx/2, nx)
y = np.linspace(dy/2, Ly - dy/2, ny)

X, Y = np.meshgrid(x, y, indexing = "ij")

T = np.ones((nx, ny)) * T_cool

x_flux = np.zeros((nx + 1, ny))
y_flux = np.zeros((nx, ny + 1))
u = np.zeros((nx + 1, ny))
v = np.zeros((nx, ny + 1))

fig, ax = plt.subplots(1, 1)
cbar = fig.colorbar(ax.contourf(X, Y, T, vmin = T_cool, vmax = T_wall_l, cmap = "hot"))

def generate_velocity_field():
    for n, y in enumerate(np.linspace(dy/2, Ly - dy/2, ny)):
        u[:, n] = (1/(2 * mu)) * p_grad * ((Ly / 2)**2 - (y - (Ly / 2))**2) #negative or not?

generate_velocity_field()


plot_counter = 0
while time < time_end:
    time += dt

    x_flux[1:-1, :] = (((T[1:, :] + T[:-1, :]) / 2) * u[1:-1, :]) - alpha * (T[1:, :] - T[:-1, :]) / dx
    y_flux[:, 1:-1] =  -alpha * (T[:, 1:] - T[:, :-1]) / dy

    x_flux[0, :] = ((T_wall_l + T[0, :]) / 2) * u[0, :] - alpha * (T[0, :] - T_wall_l) / (dx / 2.0) #
    x_flux[-1, :] = ((T[-1, :])) * u[-1, :] #- alpha * (T[-1, :] - T_wall_l) / (dx / 2.0)

    y_flux[:, 0] = 100 #1#100 * Lx
    y_flux[:, -1] = -100 #-alpha * (T_wall_t - T[:, -1]) / (dx / 2.0)

    T = T - (dt / (dx * dy)) * (dy * (x_flux[1:, :] - x_flux[:-1, :]) + dx * (y_flux[:, 1:] - y_flux[:, :-1]))


    if plot_counter % plotting == 0:
        cbar.remove()
        ax.cla()
        con = ax.contourf(X, Y, T, vmin = T_cool, vmax = 3000, cmap = "hot", levels = 15)
        cbar = fig.colorbar(con)
        ax.set_aspect("equal")

        vtk_x = np.linspace(0, Lx, nx + 1)
        vtk_y = np.linspace(0, Ly, ny + 1)

        no_slices = 1

        vtk_z = np.arange(0, no_slices + 1) * dx
        vtk_temp = np.dstack([T] * no_slices)

        gridToVTK("C:/Users/lachl/Documents/python/droplet-main/data/" + str(plot_counter).zfill(8), vtk_x, vtk_y, vtk_z, cellData = {"temp": vtk_temp})
        max_temp.append(np.max(T))

        plt.pause(0.001)
    plot_counter += 1
plt.show()

plt.plot()
