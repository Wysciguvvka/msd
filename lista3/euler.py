import math
import numpy as np


def euler_lv(interval: int, dt: float, *, x0: int, y0: int,
             a: float, b: float, c: float, d: float) -> (np.array, np.array):
    steps = math.ceil(interval / dt + 1)
    t = np.linspace(0, interval, steps)
    xy = np.zeros((2, steps))
    xy[0, 0] = x0
    xy[1, 0] = y0

    for i in range(steps - 1):
        x, y = xy[0, i], xy[1, i]
        dx = (a - b * y) * x * dt
        dy = (c * x - d) * y * dt
        xy[0, i + 1] = x + dx
        xy[1, i + 1] = y + dy
    return xy, t


def euler_lorenz(interval: int, dt: float, *, x0: int, y0: int, z0: int,
                 sigma: float, beta: float, rho: float) -> (np.array, np.array):
    steps = math.ceil(interval / dt + 1)
    t = np.linspace(0, interval, num=steps)
    xyz = np.zeros((3, steps))
    xyz[0, 0] = x0
    xyz[1, 0] = y0
    xyz[2, 0] = z0
    for i in range(steps - 1):
        x, y, z = xyz[0, i], xyz[1, i], xyz[2, i]
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = ((x * y) - (beta * z)) * dt
        xyz[0, i + 1] = x + dx
        xyz[1, i + 1] = y + dy
        xyz[2, i + 1] = z + dz
    return xyz, t
