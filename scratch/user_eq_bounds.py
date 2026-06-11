import numpy as np
from scipy.optimize import root_scalar

tau = 1.0
x_plus = (1 + np.sqrt(1 + 4/tau)) / 2
x_minus = 0.0 # or maybe (1 - np.sqrt(1+4/tau))/2 but we restrict to x>0 ? Wait. The user says "The spectral edges x_ - and x_ + correspond directly to the critical points where the discriminant vanishes".

# let's just integrate y, y/x, y/(pi x), etc on [0.01, x_plus]

def f(y, tau):
    if y == 0:
        return 1/tau
    return y / np.tan(tau * y) + y**2 / tau

x_vals = np.linspace(0.001, x_plus - 0.001, 1000)
y_vals = []
for x in x_vals:
    target = x**2 - x
    try:
        res = root_scalar(lambda y: f(y, tau) - target, bracket=[1e-6, np.pi/tau - 1e-6])
        y_vals.append(res.root)
    except:
        y_vals.append(0)

y_vals = np.array(y_vals)
import matplotlib.pyplot as plt

plt.plot(x_vals, y_vals, label="y")
plt.savefig("y_plot.png")

print("Integral y:", np.trapz(y_vals, x_vals))
print("Integral y/x:", np.trapz(y_vals / x_vals, x_vals))
print("Integral y/(pi x):", np.trapz(y_vals / (np.pi * x_vals), x_vals))
