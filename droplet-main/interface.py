import numpy as np
import matplotlib.pyplot as plt
import pandas
import json
import engine_p as DropletEngine
import tqdm

def loadJSON(
        filepath: str
    ) -> dict:

    json_data = json.loads(filepath)

    return json_data

class World:
    def __init__(self, name):
        self.name = name


    def setUpWorld(self, data: dict):
        self.Lx = data["Lx"]
        self.Ly = data["Ly"]
        self.nx = data["nx"]
        self.ny = data["ny"]

        self.gx = data["gx"]
        self.gy = data["gy"]

        self.bc = data["bc"]

    def generateBCMask(self, mask):
        self.bc_mask = mask

    def runSimulation(self, n_timesteps, total_simulation_time, plot_contour = True):
        
        print("Allocating memory...")
        x_velocity_field = np.zeros((self.nx + 1, self.ny + 2))
        y_velocity_field = np.zeros((self.nx + 2, self.ny + 1))

        x_star_field = np.zeros((self.nx + 1, self.ny + 2))
        y_star_field = np.zeros((self.nx + 2, self.ny + 1))

        x_flux_x = np.zeros((self.nx + 1, self.ny + 2))
        y_flux_x = np.zeros(((self.nx + 2, self.ny + 1)))
        x_flux_y = np.zeros((self.nx + 1, self.ny + 2))
        y_flux_y = np.zeros(((self.nx + 2, self.ny + 1)))

        dx = self.Lx / self.nx
        dy = self.Ly / self.ny

        pressure = np.zeros((self.nx + 2, self.ny + 2)) # Cell centered

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

        T_cool = 300.0 # Initial condition K
        T_wall_t = 350 # Top Wall condition K
        T_wall_l = 300.0 # Left Wall condition K

        T = np.ones((self.nx, self.ny)) * T_cool
        #T = np.ones((self.nx, self.ny)) * T_cool

        t_x_flux = np.zeros((self.nx + 1, self.ny))
        t_y_flux = np.zeros((self.nx, self.ny + 1))

        temp_bc = {}

        alpha = 0.01

        uu = np.zeros((self.nx + 1, self.ny + 1))
        vv = np.zeros((self.nx + 1, self.ny + 1))
        uold = uu.copy()
        vold = vv.copy()

        xnodes = np.linspace(dx / 2, self.Lx - dx / 2, self.nx)
        ynodes = np.linspace(dy / 2, self.Ly - dy / 2, self.ny)

        #X, Y = np.meshgrid(x, y, indexing = "ij")
        cell_vel = np.zeros((self.nx, self.ny))

        A_P = A_E + A_W + A_S + A_N

        coeffs = {"A_E": A_E, "A_W": A_W, "A_S": A_S, "A_N": A_N, "A_P": A_P}

        prhs = np.zeros_like(pressure)

        mu = 0.01 / 1
        rho = 1.0

        dt = total_simulation_time / n_timesteps

        print("Starting simulation...")
        
        steps = []

        xx, yy = np.meshgrid(xnodes, ynodes, indexing = "ij")

        if plot_contour:
            fig, ax = plt.subplots(1, 2)

        con = ax[1].contourf(xx, yy, T, vmin = 300.0, vmax = 400.0, levels = 20)
        cbar = fig.colorbar(con)
            

        for i in range(n_timesteps + 1):
            DropletEngine.applyBoundaryConditions(self.bc, x_velocity_field, y_velocity_field, x_flux_x, y_flux_x, x_flux_y, y_flux_y, bc_mask = self.bc_mask)
            
            DropletEngine.computeFluxes(T, rho, mu, self.nx, self.ny, dt, self.gx, self.gy, dx, dy, x_velocity_field, y_velocity_field, x_flux_x, y_flux_x, x_flux_y, y_flux_y, x_star_field, y_star_field)
            DropletEngine.computeTemperatureField(dt, temp_bc, self.nx, self.ny, dx, dy, x_velocity_field, y_velocity_field, t_x_flux, t_y_flux, alpha, T)
            #DropletEngine.computeStarredVelocities(dt, gx, gy, rho, mu, self.nx, self.ny, dx, dy, x_flux_x, y_flux_x, x_flux_y, y_flux_y, x_velocity_field, y_velocity_field, x_star_field, y_star_field, T, T_cool, 0.002)
            DropletEngine.correctVelocities(dt, coeffs, rho, mu, self.nx, self.ny, dx, dy, prhs, x_velocity_field, y_velocity_field, x_star_field, y_star_field, pressure)
            

            uu = 0.5 * (x_velocity_field[1:, 1:-1] + x_velocity_field[:-1, 1:-1])
            vv = 0.5 * (y_velocity_field[1:-1, 1:] + y_velocity_field[1:-1, :-1])

            steps.append(i)
            print(i)

            if plot_contour:
                if (i % 10) == 0:
                    
                    cbar.remove()
                    ax[0].contourf(xx, yy, np.sqrt(uu * uu + vv * vv), vmin = 0.0, vmax = 2.0, levels = 20)
                    con = ax[1].contourf(xx, yy, T, vmin = 300.0, vmax = 400.0, levels = 20)
                    cbar = fig.colorbar(con)
                    ax[0].quiver(xx, yy, uu, vv)
                    plt.pause(0.001)
        
        if plot_contour:
            plt.show()

        plt.show()

        


