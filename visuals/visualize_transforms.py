import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from finitefree.orthogonal import laguerre_polynomial
from finitefree.transforms import FiniteTTransform

os.makedirs("visuals/assets", exist_ok=True)
os.makedirs("visuals/assets/temp_t_frames", exist_ok=True)


def visualize_t_transform_steps():
    print(
        "Generating Fujie-Ueda Finite T-Transform step function convergence animation..."
    )
    t_grid = np.linspace(0.01, 0.99, 300)

    d_vals = [
        4,
        6,
        8,
        10,
        12,
        15,
        18,
        22,
        26,
        30,
        35,
        40,
        45,
        50,
        60,
        70,
        80,
        90,
        100,
    ]
    frame_images = []

    for idx, d in enumerate(d_vals):
        p = laguerre_polynomial(d, d)
        t_trans = FiniteTTransform(p)

        y_vals = []
        for t in t_grid:
            try:
                y_vals.append(float(t_trans(t)) / d)
            except ValueError:
                y_vals.append(0.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.step(
            t_grid,
            y_vals,
            where="post",
            color="#1f77b4",
            label=f"Finite T-Transform ($d = {d}$)",
            alpha=0.8,
            lw=2.0,
        )
        ax.plot(
            t_grid,
            1.0 + t_grid,
            "--",
            color="black",
            lw=2.0,
            label="Free MP Limit ($T(t)=1+t$)",
        )

        ax.set_title(
            "Fujie-Ueda Finite T-Transform Convergence (Laguerre/MP Ensemble)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel(r"Domain $t \in (0, 1)$")
        ax.set_ylabel("Normalized T-Transform $T_d(t) / d$")
        ax.set_ylim(0.8, 2.2)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        plt.tight_layout()

        frame_path = f"visuals/assets/temp_t_frames/frame_{idx:03d}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close()
        frame_images.append(Image.open(frame_path))

    gif_path = "visuals/assets/t_transform_convergence.gif"
    if frame_images:
        frame_images[0].save(
            gif_path,
            save_all=True,
            append_images=frame_images[1:],
            duration=200,
            loop=0,
        )
        print(f"Saved T-transform convergence animation to {gif_path}")

        # Cleanup
        for img in frame_images:
            img.close()
        for idx in range(len(d_vals)):
            frame_path = f"visuals/assets/temp_t_frames/frame_{idx:03d}.png"
            if os.path.exists(frame_path):
                os.remove(frame_path)
        os.rmdir("visuals/assets/temp_t_frames")


def visualize_cumulant_decay():
    print("Generating Asymptotic Decay of High-Order Cumulants animation...")
    import sympy as sp

    from examples.showcase import build_symmetric_roots_poly
    from finitefree.transforms import FiniteRTransform

    dimensions = [
        4,
        6,
        8,
        12,
        18,
        26,
        36,
        50,
        70,
        95,
        125,
        160,
        200,
        250,
        300,
    ]
    max_k = 10
    empirical_data = {}

    for d in dimensions:
        print(f"  Calculating for d = {d}...")
        # 1. Base expected polynomial of Bernoulli matrix of size d
        p_base = build_symmetric_roots_poly(d)

        # 2. Compute the d-th additive convolution power
        p_power = p_base.additive_power(d)

        # 3. Dilate by 1/sqrt(d)
        val = (
            sp.Rational(1, int(d**0.5))
            if int(d**0.5) ** 2 == d
            else sp.Rational(1000, int(1000 * d**0.5))
        )
        p_scaled = p_power.dilation(val)

        # 4. Extract exact cumulants
        cums = FiniteRTransform(p_scaled, order=max_k, numerical=True)
        empirical_data[d] = [abs(float(c)) for c in cums]

    # Temporary directory for frames
    temp_dir = "visuals/assets/temp_cumulant_frames"
    os.makedirs(temp_dir, exist_ok=True)
    frame_images = []

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

    k_vals = np.arange(1, max_k + 1)
    # Target limit: variance = 1.0, all other cumulants = 0
    target_cums = [0.0] * max_k
    target_cums[1] = 1.0

    for idx, d in enumerate(dimensions):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot theoretical semicircle limit in the background
        ax.plot(
            k_vals,
            target_cums,
            "--",
            color="black",
            alpha=0.4,
            linewidth=1.5,
            label="Theoretical Limit (d \u2192 \u221e)",
            zorder=2,
        )

        # Plot empirical averages as a single morphing curve
        ax.plot(
            k_vals,
            empirical_data[d],
            marker="o",
            markersize=8,
            linewidth=2.5,
            color="#FF5733",
            label=f"Finite R-Transform ($d = {d}$)",
            alpha=0.9,
            zorder=3,
        )

        ax.set_title(
            "Asymptotic Decay of High-Order Finite Free Cumulants",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Cumulant Order k", fontsize=11, labelpad=10)
        ax.set_ylabel("Cumulant Magnitude |\u03ba_k(d)|", fontsize=11, labelpad=10)

        # Log scale to capture exponential decay
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1e10)
        ax.set_xlim(0.8, max_k + 0.2)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"\u03ba_{k}" for k in k_vals], fontsize=10)

        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

        # Description box moved to top-left
        text_str = (
            f"Current Dimension: d = {d}\n"
            "\u2022 \u03ba_2 (variance) \u2192 1.0 (invariant)\n"
            "\u2022 \u03ba_k for k \u2265 3 collapse to 0"
        )
        props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.15}
        ax.text(
            0.05,
            0.95,
            text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()
        frame_path = f"{temp_dir}/frame_{idx:03d}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close()
        frame_images.append(Image.open(frame_path))

    gif_path = "visuals/assets/cumulant_decay.gif"
    if frame_images:
        frame_images[0].save(
            gif_path,
            save_all=True,
            append_images=frame_images[1:],
            duration=250,
            loop=0,
        )
        print(f"Saved cumulant decay animation to {gif_path}")

        # Cleanup
        for img in frame_images:
            img.close()
        for idx in range(len(dimensions)):
            frame_path = f"{temp_dir}/frame_{idx:03d}.png"
            if os.path.exists(frame_path):
                os.remove(frame_path)
        os.rmdir(temp_dir)


def visualize_s_transform_steps():
    print("Generating Finite S-Transform convergence animation...")
    from finitefree.convolutions import multiplicative
    from finitefree.ensembles import wishart_expected_poly
    from finitefree.transforms import FiniteSTransform

    d_vals = [
        4,
        6,
        8,
        10,
        12,
        15,
        18,
        22,
        26,
        30,
        35,
        40,
        45,
        50,
        60,
        70,
        80,
    ]

    t_grid = np.linspace(0.0, 1.0, 300)
    # The correct limits for Wishart expected polynomials are:
    # S_p(-t) = 1 / (1 - t/2)
    # S_q(-t) = 1 / (1 - t/3)
    # S_{p \boxtimes q}(-t) = 1 / ((1 - t/2) * (1 - t/3))
    s_p_limit = 1.0 / (1.0 - 0.5 * t_grid)
    s_q_limit = 1.0 / (1.0 - t_grid / 3.0)
    s_limit_vals = s_p_limit * s_q_limit

    temp_dir = "visuals/assets/temp_s_frames"
    os.makedirs(temp_dir, exist_ok=True)
    frame_images = []

    # Use a clean, modern style context
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

    for idx, d in enumerate(d_vals):
        p = wishart_expected_poly(d, 2 * d)
        q = wishart_expected_poly(d, 3 * d)
        r = multiplicative(p, q, d)

        S_p = FiniteSTransform(p, exact=False)
        S_q = FiniteSTransform(q, exact=False)
        S_r = FiniteSTransform(r, exact=False)

        t_nodes = np.arange(1, d + 1) / d

        fig, ax = plt.subplots(figsize=(8, 6))

        # 1. Plot the continuous analytical limits in the background as dashed lines
        ax.plot(
            t_grid,
            s_p_limit,
            ":",
            color="#2ca02c",
            lw=1.2,
            alpha=0.6,
            label=r"Limit $S_p(-t) = \frac{1}{1 - t/2}$",
        )
        ax.plot(
            t_grid,
            s_q_limit,
            ":",
            color="#9467bd",
            lw=1.2,
            alpha=0.6,
            label=r"Limit $S_q(-t) = \frac{1}{1 - t/3}$",
        )
        ax.plot(
            t_grid,
            s_limit_vals,
            "--",
            color="black",
            lw=1.8,
            alpha=0.8,
            label=r"Limit $S_{p \boxtimes q}(-t) = \frac{1}{(1-t/2)(1-t/3)}$",
            zorder=1,
        )

        # 2. Plot the piecewise linear interpolations of the finite S-transforms
        # Individual finite transforms as thin solid lines
        ax.plot(
            t_nodes,
            S_p,
            color="#2ca02c",
            lw=1.5,
            alpha=0.7,
            zorder=2,
            label=f"$S_p^{{({d})}}(-t)$ ($c=2$)",
        )
        ax.plot(
            t_nodes,
            S_q,
            color="#9467bd",
            lw=1.5,
            alpha=0.7,
            zorder=2,
            label=f"$S_q^{{({d})}}(-t)$ ($c=3$)",
        )

        # Convolved finite transform (which is exactly equal to the product S_p * S_q)
        ax.plot(
            t_nodes,
            S_r,
            color="#d62728",
            lw=2.0,
            alpha=0.85,
            zorder=3,
            label=f"Convolved $S_{{p \\boxtimes q}}^{{({d})}}(-t)$",
        )

        ax.set_title(
            "Finite S-Transform Multiplicativity & Convergence",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel(r"Domain $t = k/d \in (0, 1]$", fontsize=11, labelpad=8)
        ax.set_ylabel(r"Transform Value $S(-t)$", fontsize=11, labelpad=8)

        # Set proper limits since values range from 1.0 to 3.0
        ax.set_ylim(0.8, 3.2)
        ax.set_xlim(-0.02, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)

        ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

        # Text box showing the dimension and exact alignment
        text_str = (
            f"Dimension: d = {d}\n"
            "Exact Multiplicativity at all nodes:\n"
            r"$S_{p \boxtimes_d q}(-k/d) = S_p(-k/d) S_q(-k/d)$"
        )
        props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.15}
        ax.text(
            0.95,
            0.05,
            text_str,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        plt.tight_layout()
        frame_path = f"{temp_dir}/frame_{idx:03d}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close()
        frame_images.append(Image.open(frame_path))

    gif_path = "visuals/assets/s_transform_convergence.gif"
    if frame_images:
        frame_images[0].save(
            gif_path,
            save_all=True,
            append_images=frame_images[1:],
            duration=250,
            loop=0,
        )
        print(f"Saved S-transform convergence animation to {gif_path}")

        # Cleanup
        for img in frame_images:
            img.close()
        for idx in range(len(d_vals)):
            frame_path = f"{temp_dir}/frame_{idx:03d}.png"
            if os.path.exists(frame_path):
                os.remove(frame_path)
        os.rmdir(temp_dir)


def visualize_cauchy_domain_coloring():
    print("Generating Finite Cauchy Transform domain coloring animation...")
    from finitefree.ensembles import gue_expected_poly

    d_vals = [4, 6, 8, 12, 18, 26, 38, 56, 80]

    # Grid in the complex plane
    nx, ny = 600, 300
    x = np.linspace(-2.5, 2.5, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    temp_dir = "visuals/assets/temp_cauchy_frames"
    os.makedirs(temp_dir, exist_ok=True)
    frame_images = []

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

    for idx, d in enumerate(d_vals):
        p = gue_expected_poly(d)
        roots = p.evaluate_roots_float64()

        diff = Z[:, :, np.newaxis] - roots
        with np.errstate(divide="ignore", invalid="ignore"):
            G = np.mean(1.0 / diff, axis=2)

        phase = np.angle(G)

        fig, ax = plt.subplots(figsize=(8, 6))
        # Lock axes position to prevent layout shifts
        ax.set_position([0.1, 0.12, 0.85, 0.78])

        # Plot domain coloring using twilight (cyclic phase colormap) to match unitary trajectories
        ax.imshow(
            phase,
            extent=[-2.5, 2.5, -1.0, 1.0],
            cmap="twilight",
            origin="lower",
            alpha=0.95,
        )

        # Subtle pole markers (small black dots with white edge)
        ax.scatter(
            roots,
            np.zeros_like(roots),
            color="black",
            edgecolor="white",
            s=12,
            linewidths=0.6,
            zorder=5,
            label="Poles (Roots of $p_d$)",
        )

        # Background reference line for the future branch cut
        ax.plot(
            [-2.0, 2.0],
            [0.0, 0.0],
            color="white",
            linestyle="--",
            lw=1.5,
            alpha=0.5,
            label="Limiting Branch Cut $[-2, 2]$",
        )

        ax.set_title(
            "Cauchy Transform Convergence",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Re(z)", fontsize=11)
        ax.set_ylabel("Im(z)", fontsize=11)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.0, 1.0)

        ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

        # Dimension details in a text box
        text_str = f"Dimension: d = {d}"
        props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.15}
        ax.text(
            0.05,
            0.95,
            text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        frame_path = f"{temp_dir}/frame_{idx:03d}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close()
        frame_images.append(Image.open(frame_path))

    # Analytical Limit Frame (maintains exact same layout, title, and legend)
    G_limit = 0.5 * (Z - np.sqrt(Z - 2.0) * np.sqrt(Z + 2.0))
    phase_limit = np.angle(G_limit)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_position([0.1, 0.12, 0.85, 0.78])
    ax.imshow(
        phase_limit,
        extent=[-2.5, 2.5, -1.0, 1.0],
        cmap="twilight",
        origin="lower",
        alpha=0.95,
    )

    # Plot empty scatter so legend matches exactly
    ax.scatter(
        [],
        [],
        color="black",
        edgecolor="white",
        s=12,
        linewidths=0.6,
        zorder=5,
        label="Poles (Roots of $p_d$)",
    )

    # Show branch cut line (matching the legend entry)
    ax.plot(
        [-2.0, 2.0],
        [0.0, 0.0],
        color="white",
        lw=2.5,
        label="Limiting Branch Cut $[-2, 2]$",
    )

    ax.set_title(
        "Cauchy Transform Convergence",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Re(z)", fontsize=11)
    ax.set_ylabel("Im(z)", fontsize=11)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.0, 1.0)

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # Static text box showing infinity limit
    text_str = "Dimension: d = \u221e (Limit)"
    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.15}
    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    final_idx = len(d_vals)
    frame_path = f"{temp_dir}/frame_{final_idx:03d}.png"
    plt.savefig(frame_path, dpi=120)
    plt.close()
    frame_images.append(Image.open(frame_path))

    gif_path = "visuals/assets/cauchy_domain_coloring.gif"
    if frame_images:
        frame_images[0].save(
            gif_path,
            save_all=True,
            append_images=frame_images[1:],
            duration=350,
            loop=0,
        )
        print(f"Saved Cauchy domain coloring animation to {gif_path}")

        # Cleanup
        for img in frame_images:
            img.close()
        for idx in range(final_idx + 1):
            frame_path = f"{temp_dir}/frame_{idx:03d}.png"
            if os.path.exists(frame_path):
                os.remove(frame_path)
        os.rmdir(temp_dir)


if __name__ == "__main__":
    visualize_t_transform_steps()
    visualize_cumulant_decay()
    visualize_s_transform_steps()
    visualize_cauchy_domain_coloring()
