import os
import numpy as np
import matplotlib.pyplot as plt
from finitefree.orthogonal import laguerre_polynomial
from finitefree.transforms import FiniteTTransform

os.makedirs("visuals/assets", exist_ok=True)


def visualize_t_transform_steps():
    print("Generating Fujie-Ueda Finite T-Transform step function convergence...")
    t_grid = np.linspace(0.01, 0.99, 300)
    
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, d in enumerate([10, 40, 100]):
        p = laguerre_polynomial(d, d)
        t_trans = FiniteTTransform(p)
        
        y_vals = []
        for t in t_grid:
            try:
                y_vals.append(float(t_trans(t)) / d)
            except ValueError:
                y_vals.append(0.0)
                
        plt.step(t_grid, y_vals, where='post', color=colors[idx], label=f"d = {d}", alpha=0.8)
        
    plt.plot(t_grid, 1.0 + t_grid, '--', color='black', lw=2.0, label="Free MP Limit ($T(t)=1+t$)")
    
    plt.title("Fujie-Ueda Finite T-Transform Convergence (Laguerre/MP Ensemble)", fontsize=13, fontweight='bold')
    plt.xlabel(r"Domain $t \in (0, 1)$")
    plt.ylabel("Normalized T-Transform $T_d(t) / d$")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals/assets/t_transform_steps.png", dpi=150)
    plt.close()
    print("Saved plot to visuals/assets/t_transform_steps.png")


if __name__ == "__main__":
    visualize_t_transform_steps()
