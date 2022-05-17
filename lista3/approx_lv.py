from euler import euler_lv
from odeint import lv_odeint
import matplotlib.pyplot as plt
import numpy as np
import csv

A, B, C, D = (1.2, 0.6, 0.3, 0.8)
X0 = 2
Y0 = 1
TESTED_STEPS = (0.3, 0.1, 0.01)
INTERVAL = 25

fig, axs = plt.subplots(nrows=len(TESTED_STEPS), ncols=2)
print(f"{'step':>6} | {'x':>10} | {'y':>10} | {'(x+y)/2':>10}")
with open('lotkavolterra.csv', 'w+', newline='') as csvfile:

    writer = csv.writer(csvfile, delimiter=',')
    header = ['step', 'x', 'y', '(x+y)/2']
    writer.writerow(header)
    for i, dt in enumerate(TESTED_STEPS):
        eul, t = euler_lv(INTERVAL, dt, x0=X0, y0=Y0, a=A, b=B, c=C, d=D)
        ode, _ = lv_odeint(INTERVAL, dt, x0=X0, y0=Y0, a=A, b=B, c=C, d=D)
        avg_deviation_x = np.average(np.abs(eul[0] - ode[0]))
        avg_deviation_y = np.average(np.abs(eul[1] - ode[1]))
        row = [f'{dt}', f'{avg_deviation_x:.6f}', f'{avg_deviation_y:.6f}',
               f'{(avg_deviation_x + avg_deviation_y) / 2:.6f}']
        writer.writerow(row)
        print(
            f"{dt:6} | {avg_deviation_x:10.6f} | {avg_deviation_y:10.6f} | {(avg_deviation_x + avg_deviation_y) / 2:10.6f}"
        )

        axs[i, 0].plot(t, eul[1], t, eul[0])  # x,y(t)
        axs[i, 1].plot(eul[1], eul[0])  # x(y)

axs[0, 0].set_title("x,y(t)")
axs[0, 1].set_title("x(y)")
plt.suptitle(
    "Wyniki uzyskane przez metodę Eulera z dt = "
    + ",".join([str(step) for step in TESTED_STEPS])
)
plt.tight_layout()
plt.autoscale()
plt.show()

plt.plot(t, ode[0], t, ode[1])
plt.xlabel('czas')
plt.ylabel('liczba')
plt.legend(["populacja ofiar", "popualcja drapieżników"])
plt.show()