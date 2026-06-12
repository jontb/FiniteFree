import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

try:
    sys.set_int_max_str_digits(20000)
except AttributeError:
    pass

from finitefree import jacobi_polynomial
from finitefree.convolutions import multiplicative, symmetric_additive
from finitefree.core import RealRootedPolynomial
from finitefree.ensembles import gue_expected_poly, wishart_expected_poly

os.makedirs("visuals", exist_ok=True)


def visualize_wigner_semicircle() -> None:
    d = 300
    p = gue_expected_poly(d)
    res = symmetric_additive(p, p, d)
    roots = res.evaluate_roots_float64()
    scaled_roots = roots / np.sqrt(2.0)

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
    y[mask] = np.sqrt(4 - x[mask] ** 2) / (2 * np.pi)
    plt.plot(x, y, "r-", lw=2, label="Wigner Semicircle Law")

    plt.title("Wigner Semicircle Law from Finite Free Convolution")
    plt.xlabel("Scaled Eigenvalues")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("visuals/wigner_semicircle.png", dpi=150)
    plt.close()


def visualize_marchenko_pastur() -> None:
    d = 300
    c_ratio = 2.0
    p = wishart_expected_poly(d, n=int(d * c_ratio))
    roots = p.evaluate_roots_float64()
    scaled_roots = roots * c_ratio

    plt.figure(figsize=(8, 5))
    plt.hist(
        scaled_roots,
        bins=35,
        density=True,
        alpha=0.6,
        color="limegreen",
        edgecolor="black",
        label=f"Finite Roots (d={d})",
    )

    x = np.linspace(0.01, 6.0, 400)
    lambda_minus = (1 - np.sqrt(c_ratio)) ** 2
    lambda_plus = (1 + np.sqrt(c_ratio)) ** 2

    y = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    y[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (
        2 * np.pi * x[mask]
    )

    plt.plot(x, y, "r-", lw=2, label="Marchenko-Pastur Law (c=2)")

    plt.title("Marchenko-Pastur Law from Finite Free Laguerre Roots")
    plt.xlabel("Scaled Eigenvalues")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("visuals/marchenko_pastur.png", dpi=150)
    plt.close()


def visualize_interlacing_preservation() -> None:
    d = 6
    roots_p = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    roots_q = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

    p = RealRootedPolynomial.from_roots(roots_p)
    q = RealRootedPolynomial.from_roots(roots_q)

    roots_r = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    r = RealRootedPolynomial.from_roots(roots_r)

    p_add = symmetric_additive(p, r, d)
    q_add = symmetric_additive(q, r, d)

    p_mult = multiplicative(p, r, d)
    q_mult = multiplicative(q, r, d)

    roots_p_add = p_add.evaluate_roots_float64()
    roots_q_add = q_add.evaluate_roots_float64()

    roots_p_mult = p_mult.evaluate_roots_float64()
    roots_q_mult = q_mult.evaluate_roots_float64()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.scatter(
        roots_p,
        np.zeros_like(roots_p),
        color="royalblue",
        s=120,
        zorder=5,
        label="Roots of p(x)",
    )
    ax1.scatter(
        roots_q,
        np.ones_like(roots_q),
        color="crimson",
        s=120,
        zorder=5,
        label="Roots of q(x)",
    )
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(1, color="gray", linestyle="--", alpha=0.5)
    for i in range(d):
        ax1.vlines(
            roots_p[i],
            0,
            0.5,
            colors="royalblue",
            linestyles="dotted",
            alpha=0.7,
        )
        ax1.vlines(
            roots_q[i],
            0.5,
            1,
            colors="crimson",
            linestyles="dotted",
            alpha=0.7,
        )
    for i in range(d - 1):
        ax1.axvspan(roots_p[i], roots_q[i], color="lightblue", alpha=0.15, zorder=1)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["p(x)", "q(x)"])
    ax1.set_title("Original Roots: p < q")
    ax1.set_xlabel("Root Coordinate")
    ax1.legend()
    ax1.grid(True, axis="x", alpha=0.3)

    ax2.scatter(
        roots_p_add,
        np.zeros_like(roots_p_add),
        color="royalblue",
        s=120,
        zorder=5,
        label=r"Roots of $p \boxplus_6 r$",
    )
    ax2.scatter(
        roots_q_add,
        np.ones_like(roots_q_add),
        color="crimson",
        s=120,
        zorder=5,
        label=r"Roots of $q \boxplus_6 r$",
    )
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(1, color="gray", linestyle="--", alpha=0.5)
    for i in range(d):
        ax2.vlines(
            roots_p_add[i],
            0,
            0.5,
            colors="royalblue",
            linestyles="dotted",
            alpha=0.7,
        )
        ax2.vlines(
            roots_q_add[i],
            0.5,
            1,
            colors="crimson",
            linestyles="dotted",
            alpha=0.7,
        )
    for i in range(d - 1):
        ax2.axvspan(
            roots_p_add[i],
            roots_q_add[i],
            color="lightblue",
            alpha=0.15,
            zorder=1,
        )
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([r"$p \boxplus_6 r$", r"$q \boxplus_6 r$"])
    ax2.set_title(r"Additive: $p \boxplus_6 r \prec q \boxplus_6 r$")
    ax2.set_xlabel("Root Coordinate")
    ax2.legend()
    ax2.grid(True, axis="x", alpha=0.3)

    ax3.scatter(
        roots_p_mult,
        np.zeros_like(roots_p_mult),
        color="royalblue",
        s=120,
        zorder=5,
        label=r"Roots of $p \boxtimes_6 r$",
    )
    ax3.scatter(
        roots_q_mult,
        np.ones_like(roots_q_mult),
        color="crimson",
        s=120,
        zorder=5,
        label=r"Roots of $q \boxtimes_6 r$",
    )
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.axhline(1, color="gray", linestyle="--", alpha=0.5)
    for i in range(d):
        ax3.vlines(
            roots_p_mult[i],
            0,
            0.5,
            colors="royalblue",
            linestyles="dotted",
            alpha=0.7,
        )
        ax3.vlines(
            roots_q_mult[i],
            0.5,
            1,
            colors="crimson",
            linestyles="dotted",
            alpha=0.7,
        )
    for i in range(d - 1):
        ax3.axvspan(
            roots_p_mult[i],
            roots_q_mult[i],
            color="lightblue",
            alpha=0.15,
            zorder=1,
        )
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels([r"$p \boxtimes_6 r$", r"$q \boxtimes_6 r$"])
    ax3.set_title(r"Multiplicative: $p \boxtimes_6 r \prec q \boxtimes_6 r$")
    ax3.set_xlabel("Root Coordinate")
    ax3.legend()
    ax3.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("visuals/root_interlacing.png", dpi=150)
    plt.close()


def visualize_free_jacobi_arcsine() -> None:
    d = 300
    p = jacobi_polynomial(d, alpha=0, beta=0)
    roots = p.evaluate_roots_float64()

    plt.figure(figsize=(8, 5))
    plt.hist(
        roots,
        bins=35,
        density=True,
        alpha=0.6,
        color="orchid",
        edgecolor="black",
        label=f"Finite Roots (d={d})",
    )

    x = np.linspace(-0.999, 0.999, 400)
    y = 1.0 / (np.pi * np.sqrt(1.0 - x**2))
    plt.plot(x, y, "r-", lw=2, label="Arcsine Law (Free Jacobi Limit)")

    plt.title("Free Jacobi (Arcsine) Law from Finite Jacobi Roots")
    plt.xlabel("Scaled Eigenvalues / Roots")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("visuals/free_jacobi_arcsine.png", dpi=150)
    plt.close()


def visualize_free_lognormal() -> None:
    import flint

    from finitefree.convolutions import multiplicative
    from finitefree.core import RealRootedPolynomial
    from finitefree.orthogonal import laguerre_polynomial

    m = 100
    d = 100
    n = 10000
    tau = 1.0

    print("  Calculating exact polynomial roots for Free Log-Normal...")
    flint.ctx.prec = 512
    lag_poly = laguerre_polynomial(m, n - m)
    q_mn = lag_poly.dilation(flint.fmpq(1, n))

    p = RealRootedPolynomial.from_roots([1] * m)
    for _ in range(d):
        p = multiplicative(p, q_mn, m)

    roots = p.evaluate_roots_float64()

    print("  Calculating analytical Free Log-Normal curve...")
    x = np.linspace(0.01, 5.0, 500)
    eps = 1e-5
    z = x + 1j * eps

    from typing import Any

    def F(u: Any, zj: Any) -> Any:
        return (1 + u) / u * np.exp(tau * u) - zj

    def F_prime(u: Any, zj: Any) -> Any:
        return np.exp(tau * u) * (tau * u * (1 + u) - 1) / (u**2)

    analytical_roots_list: list[Any] = []
    u = -1.0 - z[0] * np.exp(tau)
    for zj in z:
        for _ in range(100):
            val = F(u, zj)
            diff = val / F_prime(u, zj)
            u -= diff
            if np.abs(val) < 1e-12:
                break
        analytical_roots_list.append(u)

    analytical_roots = np.array(analytical_roots_list)
    G = (analytical_roots + 1) / z
    density = -np.imag(G) / np.pi

    plt.figure(figsize=(8, 5))
    plt.hist(
        roots,
        bins=25,
        density=True,
        alpha=0.6,
        color="goldenrod",
        edgecolor="black",
        label=f"Finite Roots (m=d={m})",
    )
    plt.plot(x, density, "r-", lw=2, label=f"Free Log-Normal Law ($\\tau={tau}$)")

    plt.title("Free Log-Normal Law from Compound Wishart Roots")
    plt.xlabel("Eigenvalues / Roots")
    plt.ylabel("Density")
    plt.xlim(-0.2, 5.2)
    plt.ylim(0, 1.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("visuals/free_lognormal.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Generating static visualizations...")
    t_start = time.perf_counter()
    visualize_wigner_semicircle()
    visualize_marchenko_pastur()
    visualize_free_jacobi_arcsine()
    visualize_free_lognormal()
    visualize_interlacing_preservation()
    elapsed = time.perf_counter() - t_start
    print(f"Static visuals successfully generated in {elapsed:.2f} seconds.")
