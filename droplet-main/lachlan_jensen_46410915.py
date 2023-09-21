from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time as timer
import pandas as pd

def successive_over_relaxation(nx,ny,p,prhs, coeffs):
    Ae = coeffs['A_E']
    Aw = coeffs['A_W']
    An = coeffs['A_N']
    As = coeffs['A_S']
    Ap = coeffs['A_P']

    it = 0
    err = 1e5
    tol = 1e-5
    maxiter = 1000
    beta = 1.2
    while err > tol and it < maxiter:
        p_old = p.copy()
        for i in range(1,nx+1):
            ip1 = (i+1)
            im1 = (i-1) 
            if i+1>nx:
                ip1=1
            elif i-1<1:
                im1=nx
            for j in range(1,ny+1):
                p[i,j] = (1-beta)*p[i,j] + (beta/Ap[i,j]) * (Ae[i,j]*p[ip1,j] + Aw[i,j]*p[im1,j] + \
                                                             An[i,j]*p[i,j+1] + As[i,j]*p[i,j-1] - \
                                                             prhs[i,j])
        err = np.linalg.norm(p-p_old)
        it += 1
    if it == maxiter:
        print('SOR did not converge!')
    return p, err

def analytical(params, y):
    Lx = params['L']
    Ly = params['H']
    gx = params['gx']
    gy = params['gy']
    mu = params['mu']
    rho = params['rho']
    return gx*y*(Ly-y)/(2*mu)

# L2 error computation   
def L2_error(uu, yy, L2, grid, analytical_fn):
    my_sum = 0.0
    nx = uu.shape[0]
    ny = uu.shape[1]
    Ncells = nx*ny
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            y = yy[i,j]
            u_a = analytical_fn(params, y)
            my_sum += np.abs((uu[i,j] - u_a)**2)
           
            grid[i, j] = (uu[i,j] - u_a)
    l_2 = np.sqrt(my_sum/Ncells)
    L2.append(l_2)

def run_simulation(nx, ny, dt, max_steps, params):
    steps = []
    L2 = []
    Lx = params['L']
    Ly = params['H']
    gx = params['gx']
    gy = params['gy']
    mu = params['mu']
    rho = params['rho']

    nu = mu / rho # m^2 / s

    unorth, usouth = 0.0, 0.0 # m / s
    vnorth, vsouth = 0.0, 0.0 # m / s

    u = np.zeros((nx+1, ny+2))
    v = np.zeros((nx+2,ny+1)) 
    p = np.zeros((nx+2, ny+2))

    ut = np.zeros_like(u)
    vt = np.zeros_like(v) 


    dx = Lx / nx # m
    dy = Ly / ny # m
    dxdy = dx * dy

    A_E = np.ones_like(p) * (1/dx/dx)
    A_W = np.ones_like(p) * (1/dx/dx)
    A_N = np.ones_like(p) * (1/dy/dy)
    A_S = np.ones_like(p) * (1/dy/dy)

    A_N[:, -2] = 0.0
    A_S[:,1] = 0.0
    A_P = A_E + A_W + A_N + A_S

    prhs = np.zeros_like(p)

    uu = np.zeros((nx, ny))
    vv = np.zeros((nx, ny))
    uold = uu.copy()
    vold = vv.copy()
    xnodes = np.linspace(dx/2., Lx-dx/2., nx)
    ynodes = np.linspace(dy/2., Ly-dy/2., ny)

    grid = np.zeros(uu.shape)

    time = 0
    tic = timer.time()
    for step in range(max_steps):
        u[:,0] = 2.*usouth - u[:,1]
        u[:,-1] = 2.*unorth - u[:,-2]
        v[:,-1] = vnorth
        v[:,0] = vsouth
        
        for i in range(0,nx+1):
            ip1 = (i+1) % (nx+1)
            im1 = (i-1) % (nx+1)
            for j in range(1,ny+1):
                Je = 0.25*(u[ip1,j]+u[i,j])**2 - nu*(u[ip1,j]-u[i,j])/dx
                Jn = 0.25*(v[ip1,j]+v[i,j])*(u[i,j+1]+u[i,j]) - nu*(u[i,j+1]-u[i,j])/dy
                Jw = 0.25*(u[i,j]+u[im1,j])**2 - nu*(u[i,j]-u[im1,j])/dx
                Js = 0.25*(v[i,j-1]+v[ip1,j-1])*(u[i,j]+u[i,j-1]) - nu*(u[i,j]-u[i,j-1])/dy

                ut[i,j] = u[i,j] - (dt/dxdy)*(dy*(Je-Jw)+dx*(Jn-Js)) + dt*gx
                
        for i in range(1,nx+1):
            for j in range(1,ny):
                Je = 0.25*(u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j]) - nu*(v[i+1,j]-v[i,j])/dx
                Jn = 0.25*(v[i,j+1]+v[i,j])**2 - nu*(v[i,j+1]-v[i,j])/dy
                Jw = 0.25*(u[i-1,j+1]+u[i-1,j])*(v[i,j]+v[i-1,j]) - nu*(v[i,j]-v[i-1,j])/dx
                Js = 0.25*(v[i,j]+v[i,j-1])**2 - nu*(v[i,j]-v[i,j-1])/dy

                vt[i,j] = v[i,j] - (dt/dxdy)*(dy*(Je-Jw)+dx*(Jn-Js)) + dt*gy

        for i in range(0,nx+1):
            ip1 = (i+1) % (nx+1)
            im1 = (i-1) % (nx+1)
            for j in range(1,ny+1):
                prhs[i,j] = (rho/dt)*((ut[i,j]-ut[im1,j])/dx+(vt[i,j]-vt[i,j-1])/dy)

        p, err = successive_over_relaxation(nx, ny, p, prhs, {'A_E': A_E, 'A_W': A_W, 'A_N': A_N, 'A_S': A_S, 'A_P':A_P})

        for i in range(0,nx+1):
            ip1 = (i+1) % (nx+1)
            im1 = (i-1) % (nx+1)
            for j in range(1,ny+1):
                u[i,j] = ut[i,j]-dt*(1.0/dx)*(p[ip1,j]-p[i,j])/(rho)   
        
        for i in range(1,nx+1):
            for j in range(1,ny):
                v[i,j] = vt[i,j]-dt*(1.0/dy)*(p[i,j+1]-p[i,j])/(rho)
        time+=dt

        steps.append(step)

        uu = 0.5*(u[1:,1:-1] + u[:-1,1:-1])

        xx, yy = np.meshgrid(xnodes, ynodes, indexing='ij')

        L2_error(uu, yy, L2, grid, analytical)
    return uu, xx, yy, steps, L2, grid

if __name__ == '__main__':
    params = {'L': 1.0, 'H': 0.25, 'gx': 0.1, 'gy': 0.0, 'mu': 0.01, 'rho': 1.0}

    nx = 10
    ny = 20
    max_steps = 5000
    dt= 0.001

    uu, xx, yy, steps, L2, grid = run_simulation(nx, ny, dt, max_steps, params)
    fig, ax = plt.subplots(2, 1)

    fig.tight_layout(pad=3.0)

    ax[0].plot(steps, L2)
    ax[0].set_title("L2 error vs steps")
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("L2 error")

    con = ax[1].contourf(xx, yy, grid)
    fig.colorbar(con)

    ax[1].set_title("Contour plot of analytical vs numerical error")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")

    # Convergence occurs when 


    plt.show()
    