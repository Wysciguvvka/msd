import numpy as np
from scipy.integrate import odeint
import math


def lorentz(params: list, t, q, m, ex, ey, ez, bx, by, bz) -> list:
    x, y, z, vx, vy, vz = params
    delta = [
        vx,  # dx/dt
        vy,  # dy/dt
        vz,  # dz/dt
        q / m * (ex + vy * bz - vz * by),  # dvx/dt
        q / m * (ey + vz * bx - vx * bz),  # dvy/dt
        q / m * (ez + vx * by - vy * bx)  # dvz/dt
    ]
    return delta


def lorentz_odeint(interval: int, dt: float, *,
                   x0: float, y0: float, z0: float,
                   vx0: float, vy0: float, vz0: float,
                   q: float, m: float,
                   ex: float, ey: float, ez: float,
                   bx: float, by: float, bz: float,
                   ) -> (np.array, np.array):
    steps = math.ceil(interval / dt + 1)
    t = np.linspace(0, interval, steps)
    xyz0 = [x0, y0, z0, vx0, vy0, vz0]
    xyz = odeint(lorentz, xyz0, t, args=(q, m, ex, ey, ez, bx, by, bz))
    return np.transpose(xyz), t
