import numpy as np
import matplotlib.pyplot as plt

kappa = 0.41
B = 5.5

y_plus = np.linspace(0.1, 10e+4, 10**6)
u_plus_5 = np.zeros_like(y_plus)
u_plus_40 = np.zeros_like(y_plus)

for n, y_p in enumerate(y_plus):
    u_plus_5[n] = y_p
    u_plus_40[n] = 1/kappa * np.log(y_p) + B

plt.plot(y_plus, u_plus_5)
plt.plot(y_plus, u_plus_40)
plt.semilogx()
plt.ylim((0, 30))
plt.xlabel("y+")
plt.ylabel("x+")

plt.axvline(x=5, ls = "--")
plt.axvline(x=40, ls = "--")

plt.title("Layers of the boundary layer")
plt.legend(["y+: 0.01 to 5", "y+: 40+"])
plt.show()