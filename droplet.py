import numpy as np
import matplotlib.pyplot as plt
import pandas
import json
from interface import World

setup = {
    "Lx": 1.0,
    "Ly": 1.0,
    "nx": 20,
    "ny": 20,
    "gx": 0.0,
    "gy": -9.81,
    "bc": {
        "usouth": 0,
        "unorth": 2,
        "vwest": 0,
        "veast": 0,
        "vsouth": 0,
        "vnorth": 0,
        "uwest": 0,
        "ueast": 0,
        "x": "normal",
        "y": "normal"
    }
}

mask = np.loadtxt("test.txt", dtype = bool).transpose()

world = World("test_world")
world.setUpWorld(setup)
world.generateBCMask(mask)

world.runSimulation(4000, 5.0, True)







