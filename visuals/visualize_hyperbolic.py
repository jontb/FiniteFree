import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("visuals/assets", exist_ok=True)


def visualize_hyperbolic_cones():
    print("Generating Hyperbolic Eigencones plot...")
    d = 4
    np.random.seed(42)
    def rand_sym():
        X = np.random.randn(d, d)
        return (X + X.T) / np.sqrt(2 * d)
        
    A1 = np.eye(d)
    A2 = rand_sym()
    A3 = rand_sym()
    
    x_grid = np.linspace(-3.5, 3.5, 300)
    y_grid = np.linspace(-3.0, 3.0, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    for i in range(len(y_grid)):
        for j in range(len(x_grid)):
            M = X[i, j] * A1 + Y[i, j] * A2 + A3
            Z[i, j] = np.linalg.det(M)
            
    plt.figure(figsize=(9, 7))
    log_det = np.log(np.abs(Z) + 1e-4)
    cp = plt.contourf(Y, X, log_det, levels=40, cmap="viridis")
    cbar = plt.colorbar(cp)
    cbar.set_label(r"Log-Determinant magnitude $\log|\det(A(x,y,1))|$", fontsize=11)
    
    y_vals = np.linspace(-3.0, 3.0, 400)
    all_roots = []
    for y in y_vals:
        M_sub = -(y * A2 + A3)
        roots = np.linalg.eigvalsh(M_sub)
        all_roots.append(roots)
    all_roots = np.array(all_roots)
    
    for k in range(d):
        plt.plot(y_vals, all_roots[:, k], '--', color='white', lw=1.5, alpha=0.9)
                 
    plt.title("Hyperbolic Pencil Topography and Eigencones", fontsize=13, fontweight='bold')
    plt.xlabel("Coordinate $y$")
    plt.ylabel("Coordinate $x$ (Roots / Eigenvalues)")
    plt.tight_layout()
    plt.savefig("visuals/assets/hyperbolic_cones.png", dpi=150)
    plt.close()
    print("Saved plot to visuals/assets/hyperbolic_cones.png")


if __name__ == "__main__":
    visualize_hyperbolic_cones()
