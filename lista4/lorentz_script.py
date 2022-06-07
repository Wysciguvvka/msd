from lorentz import Lorentz

if __name__ == '__main__':
    m = 0.266  # *10^-27kg
    q = -3.2  # 10^-19 C

    Ex = 1  # T
    Ey = -1
    Ez = 0.2

    Bx = 0  # T
    By = 0
    Bz = 0.5

    vx0 = 2  # *10^5 m/s
    vy0 = 2
    vz0 = 1

    x0 = y0 = z0 = 0
    INTERVAL = 10
    STEPS = (0.1, 0.15, 0.2)
    opts = {'interval': INTERVAL, 'steps': STEPS}

    ics = {'x0': x0, 'y0': y0, 'z0': z0,
           'vx0': vx0, 'vy0': vy0, 'vz0': vz0}

    params = {'m': m, 'q': q,
              'ex': Ex, 'ey': Ey, 'ez': Ez,
              'bx': Bx, 'by': By, 'bz': Bz
              }

    lorentz_force = Lorentz(**ics, **params, **opts)
    lorentz_force.velocity_graph()
    lorentz_force.plot_means()
    lorentz_force.compare_graphs()
