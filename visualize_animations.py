import os

import matplotlib.pyplot as plt
import numpy as np

from finitefree.convolutions import symmetric_additive
from showcase import build_hermite_poly, build_laguerre_poly

os.makedirs("visuals", exist_ok=True)


def animate_asymptotic_convergence() -> None:
    """
    Generates an animated GIF showing the convergence of roots of He_d \boxplus_d He_d
    to the Wigner Semicircle as degree d increases from 10 to 100.
    """
    import matplotlib.animation as animation

    degrees = [10, 30, 50, 70, 100, 150, 200, 250, 300]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Pre-calculate roots using our high-precision isolation pipeline
    roots_dict = {}
    for d in degrees:
        p = build_hermite_poly(d)
        res = symmetric_additive(p, p, d)
        roots = res.evaluate_roots_float64()
        scaled_roots = roots / np.sqrt(2 * d)
        roots_dict[d] = scaled_roots

    def update(frame: int) -> None:
        ax.clear()
        d = degrees[frame]
        scaled_roots = roots_dict[d]

        # Histogram of scaled roots
        ax.hist(
            scaled_roots,
            bins=max(8, int(d / 8)),
            density=True,
            alpha=0.6,
            color="royalblue",
            edgecolor="black",
            label=f"Finite Roots (d={d})",
        )

        # Idealized Semicircle density
        x_vals = np.linspace(-2.1, 2.1, 400)
        y_vals = np.zeros_like(x_vals)
        mask = np.abs(x_vals) <= 2
        y_vals[mask] = np.sqrt(4 - x_vals[mask] ** 2) / (2 * np.pi)
        ax.plot(x_vals, y_vals, "r-", lw=2.5, label="Wigner Semicircle Law")

        ax.set_title(
            f"Wigner Semicircle Convergence: $He_d \\boxplus_d He_d$ (d={d})",
            fontsize=12,
            pad=10,
        )
        ax.set_xlabel("Scaled Eigenvalues")
        ax.set_ylabel("Density")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(0, 0.4)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=len(degrees), repeat=True)  # type: ignore[arg-type]

    gif_path = "visuals/wigner_semicircle_convergence.gif"
    try:
        ani.save(gif_path, writer="pillow", fps=2)
        print(f"Animated GIF saved successfully to {gif_path}")
    except Exception as e:
        print(f"Failed to save animation as GIF: {e}")
    plt.close()


def animate_marchenko_pastur_convergence() -> None:
    """
    Generates an animated GIF showing the convergence of roots of Laguerre polynomials
    to the Marchenko-Pastur law as degree d increases from 10 to 100.
    """
    import matplotlib.animation as animation

    degrees = [10, 30, 50, 70, 100, 150, 200, 250, 300]
    c_ratio = 2.0

    fig, ax = plt.subplots(figsize=(8, 5))

    # Pre-calculate roots using our high-precision isolation pipeline
    roots_dict = {}
    for d in degrees:
        p = build_laguerre_poly(d, c_ratio=c_ratio)
        roots = p.evaluate_roots_float64()
        scaled_roots = roots / d
        roots_dict[d] = scaled_roots

    def update(frame: int) -> None:
        ax.clear()
        d = degrees[frame]
        scaled_roots = roots_dict[d]

        # Histogram of scaled roots
        ax.hist(
            scaled_roots,
            bins=max(8, int(d / 8)),
            density=True,
            alpha=0.6,
            color="limegreen",
            edgecolor="black",
            label=f"Finite Roots (d={d})",
        )

        # Idealized Marchenko-Pastur density
        x_vals = np.linspace(0.01, 6.0, 400)
        lambda_minus = (1 - np.sqrt(c_ratio)) ** 2
        lambda_plus = (1 + np.sqrt(c_ratio)) ** 2
        y_vals = np.zeros_like(x_vals)
        mask = (x_vals >= lambda_minus) & (x_vals <= lambda_plus)
        y_vals[mask] = np.sqrt(
            (lambda_plus - x_vals[mask]) * (x_vals[mask] - lambda_minus)
        ) / (2 * np.pi * x_vals[mask])

        ax.plot(x_vals, y_vals, "r-", lw=2.5, label="Marchenko-Pastur Law (c=2)")

        ax.set_title(
            f"Marchenko-Pastur Convergence: Laguerre Roots (d={d})",
            fontsize=12,
            pad=10,
        )
        ax.set_xlabel("Scaled Eigenvalues")
        ax.set_ylabel("Density")
        ax.set_xlim(-0.2, 6.2)
        ax.set_ylim(0, 0.45)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=len(degrees), repeat=True)  # type: ignore[arg-type]

    gif_path = "visuals/marchenko_pastur_convergence.gif"
    try:
        ani.save(gif_path, writer="pillow", fps=2)
        print(f"Animated GIF saved successfully to {gif_path}")
    except Exception as e:
        print(f"Failed to save Marchenko-Pastur animation as GIF: {e}")
    plt.close()


if __name__ == "__main__":
    print("Generating Wigner Semicircle convergence animation...")
    animate_asymptotic_convergence()
    print("Generating Marchenko-Pastur convergence animation...")
    animate_marchenko_pastur_convergence()
    print("Animations successfully generated in visuals/ directory.")
