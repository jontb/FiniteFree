import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import root_scalar

os.makedirs('visuals', exist_ok=True)

def visualize_free_lognormal(tau=1.0):
    x_plus = (1 + np.sqrt(1 + 4/tau)) / 2

    def f(y):
        if y == 0:
            return 1/tau
        return y / np.tan(tau * y) + y**2 / tau

    x_vals = np.linspace(0.001, x_plus - 0.001, 400)
    y_vals = []

    for x in x_vals:
        target = x**2 - x
        try:
            res = root_scalar(lambda y: f(y) - target, bracket=[1e-6, np.pi/tau - 1e-6])
            y_vals.append(res.root)
        except:
            y_vals.append(0)

    y_vals = np.array(y_vals)

    # Normalize density
    area = np.trapz(y_vals, x_vals)
    density = y_vals / area

    plt.figure(figsize=(8, 5))

    # Free Lognormal
    plt.plot(x_vals, density, 'r-', lw=2.5, label=f"Free Lognormal ($\\tau={tau}$)")
    plt.fill_between(x_vals, 0, density, color='red', alpha=0.2)

    # Classical Lognormal
    sigma2 = np.log(1 + tau)
    mu = -sigma2 / 2
    classical_x = np.linspace(0.001, x_plus + 2, 400)
    classical_density = stats.lognorm.pdf(classical_x, s=np.sqrt(sigma2), scale=np.exp(mu))

    plt.plot(classical_x, classical_density, 'b--', lw=2, label="Classical Lognormal ($\\sigma^2=\\ln(1+\\tau)$)")

    plt.title(f"Free vs Classical Lognormal Distribution ($\\tau={tau}$)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.xlim(0, x_plus + 1)
    plt.ylim(0, max(np.max(density), np.max(classical_density)) * 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("visuals/free_lognormal.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    visualize_free_lognormal()
