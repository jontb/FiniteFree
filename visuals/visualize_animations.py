import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    sys.set_int_max_str_digits(20000)
except AttributeError:
    pass

from finitefree import jacobi_polynomial
from finitefree.convolutions import symmetric_additive
from finitefree.ensembles import gue_expected_poly, wishart_expected_poly

os.makedirs("visuals", exist_ok=True)


def animate_asymptotic_convergence() -> None:
    """
    Generates an animated GIF showing the convergence of roots of He_d \boxplus_d He_d
    to the Wigner Semicircle as degree d increases from 10 to 300.
    """
    import matplotlib.animation as animation

    degrees = [10, 30, 50, 70, 100, 150, 200, 250, 300]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Pre-calculate roots using our high-precision isolation pipeline
    roots_dict = {}
    for d in degrees:
        p = gue_expected_poly(d)
        res = symmetric_additive(p, p, d)
        roots = res.evaluate_roots_float64()
        scaled_roots = roots / np.sqrt(2.0)
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
            f"Wigner Semicircle Convergence: GUE expected poly (d={d})",
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
    to the Marchenko-Pastur law as degree d increases from 10 to 300.
    """
    import matplotlib.animation as animation

    degrees = [10, 30, 50, 70, 100, 150, 200, 250, 300]
    c_ratio = 2.0

    fig, ax = plt.subplots(figsize=(8, 5))

    # Pre-calculate roots using our high-precision isolation pipeline
    roots_dict = {}
    for d in degrees:
        p = wishart_expected_poly(d, n=int(d * c_ratio))
        roots = p.evaluate_roots_float64()
        scaled_roots = roots * c_ratio
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
            f"Marchenko-Pastur Convergence: Wishart expected poly (d={d})",
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


def animate_free_jacobi_arcsine_convergence() -> None:
    """
    Generates an animated GIF showing the convergence of roots of Jacobi (Legendre)
    polynomials to the classical Arcsine law (a degenerate case of the Free Jacobi
    distribution) as degree d increases from 10 to 300.
    """
    import matplotlib.animation as animation

    degrees = [10, 30, 50, 70, 100, 150, 200, 250, 300]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Pre-calculate roots using our high-precision isolation pipeline
    roots_dict = {}
    for d in degrees:
        p = jacobi_polynomial(d, alpha=0, beta=0)
        roots = p.evaluate_roots_float64()
        roots_dict[d] = roots

    def update(frame: int) -> None:
        ax.clear()
        d = degrees[frame]
        roots = roots_dict[d]

        # Histogram of roots
        ax.hist(
            roots,
            bins=max(8, int(d / 8)),
            density=True,
            alpha=0.6,
            color="orchid",
            edgecolor="black",
            label=f"Finite Roots (d={d})",
        )

        # Idealized Arcsine density on (-1, 1)
        # f(x) = 1 / (pi * sqrt(1 - x^2))
        x_vals = np.linspace(-0.999, 0.999, 400)
        y_vals = 1.0 / (np.pi * np.sqrt(1.0 - x_vals**2))
        ax.plot(x_vals, y_vals, "r-", lw=2.5, label="Arcsine Law (Free Jacobi Limit)")

        ax.set_title(
            f"Free Jacobi (Arcsine) Convergence: Legendre Roots (d={d})",
            fontsize=12,
            pad=10,
        )
        ax.set_xlabel("Scaled Eigenvalues / Roots")
        ax.set_ylabel("Density")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, 1.5)
        ax.legend(loc="upper center")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=len(degrees), repeat=True)  # type: ignore[arg-type]

    gif_path = "visuals/free_jacobi_arcsine_convergence.gif"
    try:
        ani.save(gif_path, writer="pillow", fps=2)
        print(f"Animated GIF saved successfully to {gif_path}")
    except Exception as e:
        print(f"Failed to save Free Jacobi animation as GIF: {e}")
    plt.close()
def animate_free_lognormal_convergence() -> None:
    """
    Generates an animated GIF showing the convergence of roots of compound Wishart
    convolutions to the analytical Free Log-Normal distribution.
    """
    import matplotlib.animation as animation
    import flint
    from finitefree.orthogonal import laguerre_polynomial
    from finitefree.convolutions import multiplicative
    from finitefree.core import RealRootedPolynomial

    degrees = [10, 20, 30, 40, 50, 60, 70, 80, 100]
    tau = 1.0

    fig, ax = plt.subplots(figsize=(8, 5))

    # Pre-calculate roots using high-precision isolation
    roots_dict = {}
    flint.ctx.prec = 512
    for m in degrees:
        d = m
        n = m * d # so that m * d / n = 1.0
        lag_poly = laguerre_polynomial(m, n - m)
        q_mn = lag_poly.dilation(flint.fmpq(1, n))
        
        p = RealRootedPolynomial.from_roots([1] * m)
        for _ in range(d):
            p = multiplicative(p, q_mn, m)
            
        roots = p.evaluate_roots_float64()
        roots_dict[m] = roots

    # Pre-calculate the exact analytical limit curve on a grid
    x_vals = np.linspace(0.01, 5.0, 500)
    eps = 1e-5
    z_vals = x_vals + 1j * eps
    
    def F(u, zj):
        return (1 + u) / u * np.exp(tau * u) - zj

    def F_prime(u, zj):
        return np.exp(tau * u) * (tau * u * (1 + u) - 1) / (u**2)

    analytical_roots = []
    u = -1.0 - z_vals[0] * np.exp(tau)
    for zj in z_vals:
        for _ in range(100):
            val = F(u, zj)
            diff = val / F_prime(u, zj)
            u -= diff
            if np.abs(val) < 1e-12:
                break
        analytical_roots.append(u)

    analytical_roots = np.array(analytical_roots)
    G = (analytical_roots + 1) / z_vals
    density = -np.imag(G) / np.pi

    def update(frame: int) -> None:
        ax.clear()
        m = degrees[frame]
        roots = roots_dict[m]

        # Histogram of roots
        ax.hist(
            roots,
            bins=max(6, int(m / 4)),
            density=True,
            alpha=0.6,
            color="goldenrod",
            edgecolor="black",
            label=f"Finite Roots (m=d={m})",
        )

        # Plot the analytical limit density
        ax.plot(x_vals, density, "r-", lw=2.5, label=f"Free Log-Normal Law ($\\tau={tau}$)")

        ax.set_title(
            f"Free Log-Normal Convergence: Compound Wishart (m=d={m}, n={m*m})",
            fontsize=12,
            pad=10,
        )
        ax.set_xlabel("Eigenvalues / Roots")
        ax.set_ylabel("Density")
        ax.set_xlim(-0.2, 5.2)
        ax.set_ylim(0, 1.8)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=len(degrees), repeat=True)  # type: ignore[arg-type]

    gif_path = "visuals/free_lognormal_convergence.gif"
    try:
        ani.save(gif_path, writer="pillow", fps=2)
        print(f"Animated GIF saved successfully to {gif_path}")
    except Exception as e:
        print(f"Failed to save Free Log-Normal animation as GIF: {e}")
    plt.close()


if __name__ == "__main__":
    print("Generating Wigner Semicircle convergence animation...")
    animate_asymptotic_convergence()
    print("Generating Marchenko-Pastur convergence animation...")
    animate_marchenko_pastur_convergence()
    print("Generating Free Jacobi Arcsine convergence animation...")
    animate_free_jacobi_arcsine_convergence()
    print("Generating Free Log-Normal convergence animation...")
    animate_free_lognormal_convergence()
    print("Animations successfully generated in visuals/ directory.")
