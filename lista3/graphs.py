from euler import euler_lorenz
from odeint import lorenz_odeint
import matplotlib.pyplot as plt

X0 = Y0 = Z0 = 1
SIGMA = 10
BETA = 8 / 3
RHO = 28
dt = 0.002
INTERVAL = 25

eul, t = euler_lorenz(INTERVAL, dt, x0=X0, y0=Y0, z0=Z0, sigma=SIGMA, beta=BETA, rho=RHO)
ode, _ = lorenz_odeint(INTERVAL, dt, x0=X0, y0=Y0, z0=Z0, sigma=SIGMA, beta=BETA, rho=RHO)

ax = plt.axes(projection='3d')
ax.plot3D(ode[0], ode[1], ode[2])
plt.show()

ax = plt.axes(projection='3d')
ax.plot3D(eul[0], eul[1], eul[2])
plt.show()

plt.plot(ode[0], ode[1])  # x(y)
plt.show()
plt.plot(ode[0], ode[2])  # z(x)
plt.show()
plt.plot(ode[1], ode[2])  # z(y)
plt.show()
