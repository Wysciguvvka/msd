import numpy as np
from scipy.integrate import odeint
import math


def lorenz(xyz, t, sigma, beta, rho):
    x, y, z = xyz
    delta = [sigma * (y - x), (x * (rho - z) - y), ((x * y) - (beta * z))]
    return delta


def lorenz_odeint(interval: int, dt: float, *, x0: int, y0: int, z0: int,
                  sigma: float, beta: float, rho: float) -> (np.array, np.array):
    steps = math.ceil(interval / dt + 1)
    t = np.linspace(0, interval, steps)
    xyz0 = [x0, y0, z0]
    xyz = odeint(lorenz, xyz0, t, args=(sigma, beta, rho))
    return np.transpose(xyz), t


def lotka_volterra(xy, t, a, b, c, d):
    prey, pred = xy
    dydt = [(a - b * pred) * prey, (c * prey - d) * pred]
    return dydt


def lv_odeint(interval: int, dt: float, *, x0: int, y0: int,
              a: float, b: float, c: float, d: float) -> (np.array, np.array):
    steps = math.ceil(interval / dt + 1)
    t = np.linspace(0, interval, steps)
    xy0 = [x0, y0]
    xy = odeint(lotka_volterra, xy0, t, args=(a, b, c, d))
    return np.transpose(xy), t
