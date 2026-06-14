import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    sys.set_int_max_str_digits(20000)
except AttributeError:
    pass

from finitefree.convolutions import multiplicative, symmetric_additive
from finitefree.core import RealRootedPolynomial

os.makedirs("visuals/assets", exist_ok=True)


def visualize_interlacing_preservation() -> None:
    print("Generating Root Interlacing Preservation plots...")
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

    # 1. Original Plot
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
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
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/assets/root_interlacing_original.png", dpi=150)
    plt.close()

    # 2. Additive Plot
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
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
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/assets/root_interlacing_additive.png", dpi=150)
    plt.close()

    # 3. Multiplicative Plot
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
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
    ax3.legend(loc="upper right")
    ax3.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/assets/root_interlacing_multiplicative.png", dpi=150)
    plt.close()
    print("Saved interlacing plots successfully.")


if __name__ == "__main__":
    visualize_interlacing_preservation()
