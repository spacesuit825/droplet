import numpy as np

nx = 20
ny = 20

arr = np.full((nx, ny), False)

np.savetxt("test.txt", arr.transpose(), fmt="%5i")