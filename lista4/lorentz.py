import math

import matplotlib.pyplot as plt
from sympy import Function, Eq, symbols, solve, \
    lambdify
from sympy.solvers.ode.systems import dsolve_system
from sympy.plotting import plot3d_parametric_line


class Lorentz:
    def __init__(self, *,
                 x0: float, y0: float, z0: float,
                 vx0: float, vy0: float, vz0: float,
                 q: float, m: float,
                 ex: float, ey: float, ez: float,
                 bx: float, by: float, bz: float,
                 steps: tuple, interval: int = 10
                 ) -> None:
        self.ics = {'x0': x0, 'y0': y0, 'z0': z0,
                    'vx0': vx0, 'vy0': vy0, 'vz0': vz0}
        self.params = {'m': m, 'q': q,
                       'ex': ex, 'ey': ey, 'ez': ez,
                       'bx': bx, 'by': by, 'bz': bz
                       }
        self.steps = steps
        self.interval = interval
        """Functions"""
        self.t = symbols('t')
        self.x = Function('x')
        self.y = Function('y')
        self.z = Function('z')
        self.vx = Function('vx')
        self.vy = Function('vy')
        self.vz = Function('vz')

    def solve_sympy(self) -> list:
        t = self.t
        x = self.x
        y = self.y
        z = self.z
        vx = self.vx
        vy = self.vy
        vz = self.vz
        m, q, ex, ey, ez, bx, by, bz = self.params.values()
        system = [
            Eq(x(t).diff(t), vx(t)),
            Eq(y(t).diff(t), vy(t)),
            Eq(z(t).diff(t), vz(t)),
            Eq(m * vx(t).diff(t), q * (ex + vy(t) * bz - vz(t) * by)),
            Eq(m * vy(t).diff(t), q * (ey + vz(t) * bx - vx(t) * bz)),
            Eq(m * vz(t).diff(t), q * (ez + vx(t) * by - vy(t) * bx))
        ]

        sol = dsolve_system(system, doit=True)
        _system = [Eq(eq.rhs.subs(t, 0), ic) for eq, ic in zip(sol[0], self.ics.values())]
        constants = solve(_system, symbols('C1 C2 C3 C4 C5 C6'))
        ans = [Eq(solution.lhs, solution.rhs.subs(constants)) for solution in sol[0]]
        # print(ans)
        return ans

    def plot_sympy(self, interval: int = None) -> None:
        if interval is None:
            interval = self.interval
        ans = self.solve_sympy()
        plot3d_parametric_line(ans[0].rhs, ans[1].rhs, ans[2].rhs, (self.t, 0, interval))

    def solve_scipy(self, steps: tuple | list = None, interval: int = None) -> list:
        from lorentz_ode import lorentz_odeint
        if interval is None:
            interval = self.interval
        if steps is None:
            steps = self.steps
        sols = []
        for dt in steps:
            ode, t = lorentz_odeint(interval, dt, **self.ics, **self.params)
            sols.append([ode, dt, t])
        return sols

    def plot_scipy(self, steps: tuple | list = None, interval: int = None) -> None:
        if interval is None:
            interval = self.interval
        if steps is None:
            steps = self.steps
        ax = plt.axes(projection='3d')
        for _sol in self.solve_scipy(steps, interval):
            sol, dt, _ = _sol
            ax.plot3D(sol[0], sol[1], sol[2], label=f'xyz(t), dt = {dt}')
        ax.legend()
        plt.tight_layout()
        plt.autoscale()
        plt.show()

    def compare_graphs(self, steps: tuple | list = None, interval: int = None) -> None:
        if interval is None:
            interval = self.interval
        if steps is None:
            steps = self.steps

        ans = self.solve_sympy()

        t = self.t
        x = self.x
        y = self.y
        z = self.z
        vx = self.vx
        vy = self.vy
        vz = self.vz

        f = lambdify(t, [ans[0].rhs, ans[1].rhs, ans[2].rhs],
                     modules=['numpy', 'sympy', {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}])

        timestamp = [n * 0.0001 for n in range(interval * 10000)]
        xyz = [f(_x) for _x in timestamp]
        ax = plt.axes(projection='3d')

        for _sol in self.solve_scipy(steps, interval):
            sol, dt, _ = _sol
            ax.plot3D(sol[0], sol[1], sol[2], label=f'xyz(t), dt = {dt}')

        ax.plot3D(*zip(*xyz), label='xyz(t) Sympy Solution')
        ax.legend()
        plt.tight_layout()
        plt.autoscale()
        plt.show()

    def velocity_graph(self, steps: tuple | list = None, interval: int = None) -> None:
        import numpy as np
        if interval is None:
            interval = self.interval
        if steps is None:
            steps = self.steps
        ans = self.solve_sympy()

        t = self.t
        x = self.x
        y = self.y
        z = self.z
        vx = self.vx
        vy = self.vy
        vz = self.vz

        f = lambdify(t, [ans[3].rhs, ans[4].rhs, ans[5].rhs],
                     modules=['numpy', 'sympy', {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}])

        timestamp = [n * 0.0001 for n in range(interval * 10000)]
        xyz = [f(_x) for _x in timestamp]
        vxyz = [list(i) for i in zip(*xyz)]
        ax = plt.axes()
        scipy_sol = self.solve_scipy(steps, interval)
        for _sol in scipy_sol:
            sol, dt, _ = _sol
            _steps = math.ceil(interval / dt + 1)
            _timestamp = np.linspace(0, interval, _steps)
            ax.plot(_timestamp, sol[3], label=f'vx(t), dt = {dt}')
        ax.plot(timestamp, vxyz[0], label=f'vx(t)')
        ax.legend()
        plt.tight_layout()
        plt.autoscale()
        plt.show()
        plt.close()

        ax = plt.axes()
        for _sol in scipy_sol:
            sol, dt, _ = _sol
            _steps = math.ceil(interval / dt + 1)
            _timestamp = np.linspace(0, interval, _steps)
            ax.plot(_timestamp, sol[4], label=f'vy(t), dt = {dt}')
        ax.plot(timestamp, vxyz[1], label=f'vy(t)')
        ax.legend()
        plt.tight_layout()
        plt.autoscale()
        plt.show()
        plt.close()

        ax = plt.axes()
        for _sol in scipy_sol:
            sol, dt, _ = _sol
            _steps = math.ceil(interval / dt + 1)
            _timestamp = np.linspace(0, interval, _steps)
            ax.plot(_timestamp, sol[5], label=f'vz(t), dt = {dt}')
        ax.plot(timestamp, vxyz[2], label=f'vz(t)')
        ax.legend()
        plt.tight_layout()
        plt.autoscale()
        plt.show()

    def mean_errors(self, steps: tuple | list = None, interval: int = None) -> tuple[dict, dict]:
        if interval is None:
            interval = self.interval
        if steps is None:
            steps = self.steps

        sympy_solutions = []
        for equation in self.solve_sympy():
            sympy_solutions.append(equation.rhs)
        scipy_solutions = self.solve_scipy(steps, interval)

        mean_abs = {}
        mean_squared = {}
        for _sol in scipy_solutions:
            sol, dt, timestamp = _sol
            avg_mean = []
            sqr_mean = []
            for step, _t in enumerate(timestamp):
                exact_values = []
                approx_values = []
                for approx_value in sol:
                    approx_values.append(approx_value[step])
                for exact_function in sympy_solutions:
                    exact_values.append(exact_function.subs(self.t, _t))
                avg_mean.append(sum([abs(exact - approx) for exact, approx in zip(exact_values, approx_values)])
                                / (step + 1))
                sqr_mean.append(sum([pow(exact - approx, 2) for exact, approx in zip(exact_values, approx_values)])
                                / (step + 1))
            mean_abs[str(dt)] = [avg_mean, timestamp.tolist()]
            mean_squared[str(dt)] = [sqr_mean, timestamp.tolist()]
        return mean_abs, mean_squared

    def plot_means(self, steps: tuple | list = None, interval: int = None) -> None:
        if interval is None:
            interval = self.interval
        if steps is None:
            steps = self.steps
        mean_abs, mean_squared = self.mean_errors(steps, interval)
        fig, axes = plt.subplots(nrows=1, ncols=1)
        """for ax, col in zip(axes[0], steps):
            ax.set_title(col)"""
        for step, means in mean_abs.items():
            axes.scatter(means[1], means[0], label=f'dt = {step}')
        axes.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        fig, axes = plt.subplots(nrows=1, ncols=1)
        for step, means in mean_squared.items():
            axes.scatter(means[1], means[0], label=f'dt = {step}')
        axes.legend()
        plt.tight_layout()
        plt.show()
