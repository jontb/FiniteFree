import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar


def f(y, tau):
    if y == 0:
        return 1/tau
    return y * 1/np.tan(tau * y) + y**2 / tau

tau = 1.0
x_vals = np.linspace(0.01, 3.0, 500)
y_vals = []

for x in x_vals:
    target = x**2 - x
    if target > 1/tau:
        y_vals.append(0)
        continue
    # find y in (0, pi/tau) such that f(y) = target
    try:
        res = root_scalar(lambda y: f(y, tau) - target, bracket=[1e-6, np.pi/tau - 1e-6])
        y_vals.append(res.root)
    except Exception:
        y_vals.append(0)

y_vals = np.array(y_vals)

# Try different densities
plt.figure()
plt.plot(x_vals, y_vals, label='y')
plt.plot(x_vals, y_vals / (np.pi * x_vals), label='y / (pi x)')
plt.plot(x_vals, y_vals / np.pi, label='y / pi')
plt.legend()
plt.title(f'tau={tau}')
plt.savefig('test_lognormal.png')
print("Integral of y / (pi x):", np.trapz(y_vals / (np.pi * x_vals), x_vals))
print("Integral of y / pi:", np.trapz(y_vals / np.pi, x_vals))
