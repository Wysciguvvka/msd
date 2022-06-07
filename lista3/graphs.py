from euler import euler_lorenz, euler_lv
from odeint import lorenz_odeint, lv_odeint
import matplotlib.pyplot as plt


class LorenzAttractor:
    X0 = Y0 = Z0 = 1
    SIGMA = 10
    BETA = 8 / 3
    RHO = 28
    dt = 0.002
    INTERVAL = 25


class LotkaVolterra:
    A, B, C, D = (1.2, 0.6, 0.3, 0.8)
    X0 = 2
    Y0 = 1
    TESTED_STEPS = (0.3, 0.1, 0.01)
    dt = 0.002
    INTERVAL = 25


eul, t = euler_lorenz(LorenzAttractor.INTERVAL, LorenzAttractor.dt,
                      x0=LorenzAttractor.X0, y0=LorenzAttractor.Y0,
                      z0=LorenzAttractor.Z0, sigma=LorenzAttractor.SIGMA,
                      beta=LorenzAttractor.BETA, rho=LorenzAttractor.RHO)
ode, _ = lorenz_odeint(LorenzAttractor.INTERVAL, LorenzAttractor.dt,
                       x0=LorenzAttractor.X0, y0=LorenzAttractor.Y0, z0=LorenzAttractor.Z0,
                       sigma=LorenzAttractor.SIGMA, beta=LorenzAttractor.BETA, rho=LorenzAttractor.RHO)

eul_lv, t_lv = euler_lv(LotkaVolterra.INTERVAL, LotkaVolterra.dt,
                        x0=LotkaVolterra.X0, y0=LotkaVolterra.Y0,
                        a=LotkaVolterra.A, b=LotkaVolterra.B, c=LotkaVolterra.C, d=LotkaVolterra.D)
ode_lv, _ = lv_odeint(LotkaVolterra.INTERVAL, LotkaVolterra.dt,
                      x0=LotkaVolterra.X0, y0=LotkaVolterra.Y0,
                      a=LotkaVolterra.A, b=LotkaVolterra.B, c=LotkaVolterra.C, d=LotkaVolterra.D)

ax = plt.axes(projection='3d')
ax.plot3D(ode[0], ode[1], ode[2])
plt.show()

ax = plt.axes(projection='3d')
ax.plot3D(eul[0], eul[1], eul[2])
plt.show()

plt.plot(ode[0], ode[1])  # x(y)
plt.plot(eul[0], eul[1])  # x(y)
plt.legend(['Odeint', 'Metoda Eulera'])
plt.show()
plt.plot(ode[0], ode[2])  # z(x)
plt.plot(eul[0], eul[2])  # z(x)
plt.legend(['Odeint', 'Metoda Eulera'])
plt.show()
plt.plot(ode[1], ode[2])  # z(y)
plt.plot(eul[1], eul[2])  # z(y)
plt.legend(['Odeint', 'Metoda Eulera'])
plt.show()

# lotka volterra

plt.plot(t_lv, ode_lv[0], t_lv, ode_lv[1])
plt.plot(t_lv, eul_lv[0], t_lv, eul_lv[1])
plt.legend(['Odeint x(t)', 'Odeint y(t)', 'Metoda Eulera x(t)', 'Metoda Eulera y(t)'])
plt.show()
