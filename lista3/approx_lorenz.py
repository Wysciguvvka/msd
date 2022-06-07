from euler import euler_lorenz
from odeint import lorenz_odeint
import matplotlib.pyplot as plt
import numpy as np
import csv

X0 = Y0 = Z0 = 1
SIGMA = 10
BETA = 8 / 3
RHO = 28
TESTED_STEPS = (0.03, 0.02, 0.01)
INTERVAL = 25

fig, axs = plt.subplots(nrows=len(TESTED_STEPS), ncols=3)
print(f"{'step':>6} | {'x':>10} | {'y':>10} | {'z':>10} | {'(x+y+z)/3':>10}")


with open('lorenz.csv', 'w+', newline='') as csvfile:

    writer = csv.writer(csvfile, delimiter=',')
    header = ['step', 'x', 'y', 'z', '(x+y+z)/3']
    writer.writerow(header)
    for i, dt in enumerate(TESTED_STEPS):
        eul, t = euler_lorenz(INTERVAL, dt, x0=X0, y0=Y0, z0=Z0, sigma=SIGMA, beta=BETA, rho=RHO)
        ode, _ = lorenz_odeint(INTERVAL, dt, x0=X0, y0=Y0, z0=Z0, sigma=SIGMA, beta=BETA, rho=RHO)
        avg_deviation_x = np.average(np.abs(eul[0] - ode[0]))
        avg_deviation_y = np.average(np.abs(eul[1] - ode[1]))
        avg_deviation_z = np.average(np.abs(eul[2] - ode[2]))
        row = [f'{dt}', f'{avg_deviation_x:.6f}', f'{avg_deviation_y:.6f}', f'{avg_deviation_z:.6f}', f'{(avg_deviation_x + avg_deviation_y + avg_deviation_z) / 3:.6f}']
        writer.writerow(row)
        print(
            f"{dt:6} | {avg_deviation_x:10.6f} | {avg_deviation_y:10.6f} | {avg_deviation_z:10.6f} | {(avg_deviation_x + avg_deviation_y + avg_deviation_z) / 3:10.6f}"
        )

        axs[i, 0].plot(eul[1], eul[0])  # x(y)
        axs[i, 1].plot(eul[2], eul[1])  # y(z)
        axs[i, 2].plot(eul[2], eul[0])  # x(z)

axs[0, 0].set_title("x(y)")
axs[0, 1].set_title("y(z)")
axs[0, 2].set_title("x(z)")
plt.suptitle(
    "Wyniki uzyskane przez metodę Eulera z dt = "
    + ",".join([str(step) for step in TESTED_STEPS])
)
plt.tight_layout()
plt.autoscale()
plt.show()
