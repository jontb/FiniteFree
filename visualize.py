import os

import matplotlib.pyplot as plt
import numpy as np

from finitefree.convolutions import symmetric_additive
from finitefree.core import RealRootedPolynomial
from showcase import build_hermite_poly, build_laguerre_poly

os.makedirs('visuals', exist_ok=True)

def visualize_wigner_semicircle() -> None:
    d = 300
    p = build_hermite_poly(d)
    # Wigner Semicircle convolution
    res = symmetric_additive(p, p, d)

    # Use our high-precision isolation pipeline to avoid Wilkinson's phenomenon
    roots = res.evaluate_roots_float64()

    # Scale roots: variance of He_d \boxplus He_d is 2d, so scale by 1/sqrt(2d)
    # Variance of He_d is d. res variance is 2d. Standard deviation is sqrt(2d).
    # Semicircle radius is R = 2. Variance of semicircle is R^2 / 4 = 1.
    # So we should scale roots by 1 / sqrt(2d)
    scaled_roots = roots / np.sqrt(2 * d)

    plt.figure(figsize=(8, 5))
    plt.hist(
        scaled_roots,
        bins=35,
        density=True,
        alpha=0.6,
        color="blue",
        edgecolor="black",
        label=f"Finite Roots (d={d})",
    )

    x = np.linspace(-2.1, 2.1, 400)
    y = np.zeros_like(x)
    mask = np.abs(x) <= 2
    y[mask] = np.sqrt(4 - x[mask]**2) / (2 * np.pi)
    plt.plot(x, y, 'r-', lw=2, label='Wigner Semicircle Law')

    plt.title('Wigner Semicircle Law from Finite Free Convolution')
    plt.xlabel('Scaled Eigenvalues')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visuals/wigner_semicircle.png', dpi=150)
    plt.close()

def visualize_marchenko_pastur() -> None:
    d = 300
    c_ratio = 2.0
    p = build_laguerre_poly(d, c_ratio=c_ratio)

    # Use our high-precision isolation pipeline to avoid Wilkinson's phenomenon
    roots = p.evaluate_roots_float64()

    # Laguerre polynomial roots for L_d^(alpha)(x)
    # Expected scaling for Marchenko-Pastur is 1/d
    scaled_roots = roots / d

    plt.figure(figsize=(8, 5))
    plt.hist(
        scaled_roots,
        bins=35,
        density=True,
        alpha=0.6,
        color="green",
        edgecolor="black",
        label=f"Finite Roots (d={d})",
    )

    x = np.linspace(0.01, 6.0, 400)
    lambda_minus = (1 - np.sqrt(c_ratio))**2
    lambda_plus = (1 + np.sqrt(c_ratio))**2

    y = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    y[mask] = np.sqrt(
        (lambda_plus - x[mask]) * (x[mask] - lambda_minus)
    ) / (2 * np.pi * x[mask])

    plt.plot(x, y, 'r-', lw=2, label='Marchenko-Pastur Law (c=2)')

    plt.title('Marchenko-Pastur Law from Finite Free Laguerre Roots')
    plt.xlabel('Scaled Eigenvalues')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visuals/marchenko_pastur.png', dpi=150)
    plt.close()

def visualize_root_interlacing() -> None:
    # Degree 8 is perfect for clear visual separation
    d = 8
    p = build_hermite_poly(d)

    # Calculate roots of p(x)
    roots_p = np.sort(np.real(np.roots(np.array(p.coeffs, dtype=float))))

    # Derivative coefficients
    dp_coeffs = []
    for k in range(d):
        dp_coeffs.append(p.coeffs[k] * (d - k))
    roots_dp = np.sort(np.real(np.roots(np.array(dp_coeffs, dtype=float))))

    plt.figure(figsize=(10, 4))

    # Plot the roots on two separate levels
    plt.scatter(
        roots_p,
        np.zeros_like(roots_p),
        color="royalblue",
        s=120,
        zorder=5,
        label=f"Roots of p(x) (d={d})",
    )
    plt.scatter(
        roots_dp,
        np.ones_like(roots_dp),
        color="crimson",
        s=120,
        zorder=5,
        label="Roots of p'(x)",
    )

    # Draw horizontal guide lines
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(1, color="gray", linestyle="--", alpha=0.5)

    # Draw vertical connecting lines or shaded regions showing interlacing intervals
    # Between roots_p[i] and roots_p[i+1]
    for i in range(d - 1):
        plt.axvspan(roots_p[i], roots_p[i + 1], color="lightblue", alpha=0.15, zorder=1)
        # Draw a line from root of dp to its boundaries
        plt.vlines(
            roots_dp[i],
            0,
            1,
            colors="crimson",
            linestyles="dotted",
            alpha=0.7,
            zorder=2,
        )

    plt.ylim(-0.5, 1.5)
    plt.yticks([0, 1], ["p(x) Roots", "p'(x) Roots"])
    plt.title("Visual Proof of Root Interlacing Geometry")
    plt.xlabel("Root Coordinate on Real Axis")
    plt.legend()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/root_interlacing.png", dpi=150)
    plt.close()


def visualize_complexity_benchmark() -> None:
    import time

    # Degrees for both convolutions and eager validation
    degrees = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    times_add = []
    for d in degrees:
        p = build_hermite_poly(d)
        q = build_hermite_poly(d)
        t0 = time.perf_counter()
        symmetric_additive(p, q, d)
        times_add.append(time.perf_counter() - t0)

    times_eager = []
    for d in degrees:
        p = build_hermite_poly(d)
        poly = RealRootedPolynomial(p.coeffs, assume_real_rooted=False)
        t0 = time.perf_counter()
        poly.verify_real_rootedness()
        times_eager.append(time.perf_counter() - t0)

    plt.figure(figsize=(8, 6))

    # Plot empirical times
    plt.loglog(
        degrees,
        times_add,
        "o-",
        color="royalblue",
        lw=2.5,
        ms=8,
        label=r"Symmetric Additive Convolution ($\boxplus_d$)",
    )
    plt.loglog(
        degrees,
        times_eager,
        "s--",
        color="crimson",
        lw=2.5,
        ms=8,
        label="Eager Sturm Validation",
    )

    # Draw theoretical guide lines
    # Anchor theoretical curves at the last data point to align slopes beautifully
    ref_d2 = [times_add[-1] * (d / degrees[-1]) ** 2 for d in degrees]
    plt.loglog(
        degrees,
        ref_d2,
        "k:",
        alpha=0.6,
        lw=1.5,
        label=r"Theoretical $O(d^2)$ Scaling",
    )

    ref_d3 = [times_eager[-1] * (d / degrees[-1]) ** 3 for d in degrees]
    plt.loglog(
        degrees,
        ref_d3,
        "r:",
        alpha=0.6,
        lw=1.5,
        label=r"Theoretical $O(d^3)$ Scaling",
    )

    plt.title("Log-Log Complexity Benchmark: Convolutions vs. Eager Validation")
    plt.xlabel("Polynomial Degree (d)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("visuals/complexity_benchmark.png", dpi=150)
    plt.close()


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
        # Pillow is standard and usually available
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
    print("Generating static visualizations...")
    visualize_wigner_semicircle()
    visualize_marchenko_pastur()
    visualize_root_interlacing()
    visualize_complexity_benchmark()
    print("Generating Wigner Semicircle convergence animation...")
    animate_asymptotic_convergence()
    print("Generating Marchenko-Pastur convergence animation...")
    animate_marchenko_pastur_convergence()
    print("Visuals and animations successfully generated in visuals/ directory.")


