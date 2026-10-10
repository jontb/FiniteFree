"""Bounded CPU showcase of GUE interval-count concentration; no new library API.

OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python examples/gue_concentration.py \
    --output /tmp/gue-concentration.json --plot-prefix /tmp/gue-concentration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import flint
import numpy as np
import scipy
import sympy as sp
from scipy.special import betaincinv, roots_legendre

from finitefree import (
    OrthogonalPolynomialKernel,
    gap_probability_continuous,
    gue_expected_poly,
    sample_gue,
)

SOURCES = {
    "count_law": "https://arxiv.org/pdf/math/0503110 (Theorem 7; restriction to A)",
    "op_recurrence": "https://dlmf.nist.gov/18.2",
    "binomial_intervals": "Clopper-Pearson, two-sided pointwise 95%; zero-hit one-sided 95% upper limit",
    "bernstein": "Independent centered Bernoulli summands have absolute bound 1; two-sided Bernstein: min(1, 2 exp(-t^2/(2v+2t/3)))",
}


def normalized_basis(n: int, points: Any) -> Any:
    """Example-local orthonormal Hermite values under N(0,1/n), degrees 0:n.

    This vectorized recurrence supplies the Gram factors. The library's exact-
    verified projected-polynomial kernel independently checks their diagonal sum.
    """
    values = np.empty((len(points), n))
    values[:, 0] = 1.0
    previous: Any = np.zeros(len(points))
    for j in range(n - 1):
        values[:, j + 1] = (
            points * math.sqrt(n / (j + 1)) * values[:, j]
            - math.sqrt(j / (j + 1)) * previous
        )
        previous = values[:, j]
    return values


def restricted_gram(n: int, interval: tuple[float, float], order: int) -> Any:
    """Positive-weight Gauss-Legendre approximation to integral_A phi phi^T dmu."""
    a, b = interval
    if n < 1 or order < 1 or not np.isfinite([a, b]).all() or a >= b:
        raise ValueError(
            "Require positive dimension/order and a finite ordered interval"
        )
    nodes, weights = roots_legendre(order)
    points = (b - a) * nodes / 2 + (a + b) / 2
    weights = (
        weights * (b - a) / 2 * np.sqrt(n / (2 * np.pi)) * np.exp(-n * points**2 / 2)
    )
    weighted = normalized_basis(n, points) * np.sqrt(weights[:, None])
    return weighted.T @ weighted


def count_law(gram: Any) -> dict[str, Any]:
    """Numerical finite-rank count law; reject material spectral violations."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not np.isfinite(gram).all():
        raise ValueError("Gram matrix must be finite and square")
    symmetry = float(np.max(np.abs(gram - gram.T), initial=0.0))
    if symmetry > 1e-12:
        raise ValueError("Gram matrix must be symmetric")
    raw = np.linalg.eigvalsh(gram)
    if np.min(raw, initial=0.0) < -1e-12 or np.max(raw, initial=1.0) > 1 + 1e-12:
        raise ValueError(
            "Restricted Gram spectrum must lie in [0,1]; refine quadrature"
        )
    eigenvalues = np.clip(raw, 0.0, 1.0)
    pmf = np.array([1.0])
    for probability in eigenvalues:
        pmf = np.convolve(pmf, [1 - probability, probability])
    mass_before_normalization = float(np.sum(pmf))
    pmf /= mass_before_normalization
    return {
        "pmf_mass_before_normalization": mass_before_normalization,
        "mean_count": float(np.sum(eigenvalues)),
        "variance_count": float(np.sum(eigenvalues * (1 - eigenvalues))),
        "trace_mean": float(np.trace(gram)),
        "trace_variance": float(np.trace(gram) - np.sum(gram * gram.T)),
        "symmetry_error": symmetry,
        "raw_eigenvalue_min": float(raw.min()),
        "raw_eigenvalue_max": float(raw.max()),
        "roundoff_clip_max": float(np.max(np.abs(raw - eigenvalues), initial=0.0)),
        "eigenvalues": eigenvalues.tolist(),
        "pmf": pmf.tolist(),
    }


def interval_counts(eigenvalues: Any, interval: tuple[float, float]) -> Any:
    """Each row is one independent matrix; endpoints are included."""
    return np.count_nonzero(
        (eigenvalues >= interval[0]) & (eigenvalues <= interval[1]), axis=-1
    )


def sample_counts(
    n: int, samples: int, seed: int, interval: tuple[float, float]
) -> Any:
    if n < 1 or not 2 <= samples <= 2000:
        raise ValueError("Require positive dimension and 2 <= samples <= 2000")
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return np.array(
            [
                int(interval_counts(np.linalg.eigvalsh(sample_gue(n)), interval))
                for _ in range(samples)
            ],
            dtype=int,
        )
    finally:
        np.random.set_state(state)


def binomial_interval(hits: int, samples: int) -> tuple[float, float]:
    """Two-sided pointwise 95% Clopper-Pearson interval, including zero/all hits."""
    if samples < 1 or not 0 <= hits <= samples:
        raise ValueError("Require 0 <= hits <= samples and samples > 0")
    lower = 0.0 if hits == 0 else float(betaincinv(hits, samples - hits + 1, 0.025))
    upper = (
        1.0 if hits == samples else float(betaincinv(hits + 1, samples - hits, 0.975))
    )
    return lower, upper


def bernstein(n: int, variance_count: float, epsilon: float) -> float:
    if epsilon == 0:
        return 1.0
    t = n * epsilon
    return min(1.0, 2 * math.exp(-t * t / (2 * variance_count + 2 * t / 3)))


def theory_case(n: int, interval: tuple[float, float], order: int) -> dict[str, Any]:
    coarse = restricted_gram(n, interval, order)
    fine = restricted_gram(n, interval, 2 * order)
    low, law = count_law(coarse), count_law(fine)
    # Compose the actual existing expected-polynomial -> projection -> kernel path.
    p = gue_expected_poly(n)
    kernel = OrthogonalPolynomialKernel(
        [p.projection(j) for j in range(n + 1)],
        [sp.factorial(j) / sp.Integer(n) ** j for j in range(n)],
    )
    nodes, weights = roots_legendre(2 * order)
    a, b = interval
    points = (b - a) * nodes / 2 + (a + b) / 2

    def weight(x: Any) -> Any:
        return np.sqrt(n / (2 * np.pi)) * np.exp(-n * x * x / 2)

    mean_from_kernel = float(
        np.sum(
            weights
            * (b - a)
            / 2
            * weight(points)
            * np.array([kernel(x, x) for x in points])
        )
    )
    # The full-window gap is too small to resolve from eigenvalues near one.
    # Use a separately labelled microscopic interval with a resolvable gap.
    gap_interval = (-0.5 / n, 0.5 / n)
    gap_gram = restricted_gram(n, gap_interval, 2 * order)
    gap_det = float(np.linalg.det(np.eye(n) - gap_gram))
    gap_nystrom = gap_probability_continuous(
        kernel, *gap_interval, n_points=48, weight_func=weight
    )
    return {
        **law,
        "mean_fraction": law["mean_count"] / n,
        "variance_fraction": law["variance_count"] / n**2,
        "quadrature_orders": [order, 2 * order],
        "gram_refinement_max_abs": float(np.max(np.abs(fine - coarse))),
        "mean_count_refinement_abs": abs(law["mean_count"] - low["mean_count"]),
        "variance_count_refinement_abs": abs(
            law["variance_count"] - low["variance_count"]
        ),
        "pmf_refinement_l1": float(np.sum(np.abs(np.asarray(law["pmf"]) - low["pmf"]))),
        "kernel_integrated_count_mean": mean_from_kernel,
        "kernel_trace_disagreement_abs": abs(mean_from_kernel - law["mean_count"]),
        "gap_check_interval": list(gap_interval),
        "gap_gram_determinant": gap_det,
        "gap_existing_nystrom": gap_nystrom,
        "gap_disagreement_abs": abs(gap_det - gap_nystrom),
        "expected_polynomial_root_fraction": float(
            np.mean(
                (p.evaluate_roots_float64() >= a) & (p.evaluate_roots_float64() <= b)
            )
        ),
    }


def run_showcase(
    dimensions: list[int], samples: int, seed: int, order: int = 96
) -> dict[str, Any]:
    if (
        not dimensions
        or min(dimensions) < 1
        or max(dimensions) > 64
        or not 2 <= samples <= 2000
        or not 8 <= order <= 256
    ):
        raise ValueError(
            "Use dimensions 1..64, samples 2..2000 and quadrature order 8..256"
        )
    interval = (-0.5, 0.5)
    records = []
    for n in dimensions:
        # Pilot draws are excluded from the fixed-size experiment and use a
        # separate deterministic seed. Do not stop sampling based on outcomes.
        pilot_seed = seed + 10000 + n
        started = perf_counter()
        sample_counts(n, 20, pilot_seed, interval)
        pilot_seconds = perf_counter() - started
        started = perf_counter()
        theory = theory_case(n, interval, order)
        theory_seconds = perf_counter() - started
        sampling_seed = seed + n
        started = perf_counter()
        counts = sample_counts(n, samples, sampling_seed, interval)
        sample_seconds = perf_counter() - started
        fractions = counts / n
        mean = theory["mean_fraction"]
        centered_square = (fractions - mean) ** 2
        # Use the known numerical mean for an unbiased variance estimator. Its
        # standard error is computed across independent matrix observations.
        variance_estimate = float(np.mean(centered_square))
        variance_se = float(np.std(centered_square, ddof=1) / math.sqrt(samples))
        mean_se = float(np.std(fractions, ddof=1) / math.sqrt(samples))
        support = np.arange(n + 1) / n
        pmf = np.asarray(theory["pmf"])
        tails = []
        for epsilon in np.linspace(0.0, 0.16, 17):
            hits = int(np.count_nonzero(np.abs(fractions - mean) >= epsilon))
            lower, upper = binomial_interval(hits, samples)
            probability = min(
                1.0, float(np.sum(pmf[np.abs(support - mean) >= epsilon]))
            )
            if epsilon == 0:
                probability = 1.0
            tails.append(
                {
                    "epsilon": float(epsilon),
                    "hits": hits,
                    "empirical_probability": hits / samples,
                    "pointwise_cp95": [lower, upper],
                    "zero_hit_one_sided_95_upper": -math.expm1(math.log(0.05) / samples)
                    if hits == 0
                    else None,
                    "numerical_count_law_tail": probability,
                    "numerical_bernstein_reference": bernstein(
                        n, theory["variance_count"], float(epsilon)
                    ),
                    "theory_outside_pointwise_interval": not lower
                    <= probability
                    <= upper,
                    "threshold_distance_to_count_support": float(
                        np.min(np.abs(np.abs(support - mean) - epsilon))
                    ),
                }
            )
        records.append(
            {
                "dimension": n,
                "samples": samples,
                "sampling_seed": sampling_seed,
                "pilot_seed": pilot_seed,
                "pilot_samples": 20,
                "pilot_seconds": pilot_seconds,
                "theory_seconds": theory_seconds,
                "sample_seconds": sample_seconds,
                "theory": theory,
                "observed_counts": counts.tolist(),
                "sample_mean_fraction": float(np.mean(fractions)),
                "mean_standard_error": mean_se,
                "sample_variance_about_theory_mean": variance_estimate,
                "variance_standard_error": variance_se,
                "sample_variance_unbiased": float(np.var(fractions, ddof=1)),
                "mean_disagreement": float(np.mean(fractions) - mean),
                "variance_disagreement": variance_estimate
                - theory["variance_fraction"],
                "tails": tails,
            }
        )
    root = Path(__file__).resolve().parents[1]
    return {
        "interval": list(interval),
        "dimensions": dimensions,
        "samples_per_dimension": samples,
        "seed": seed,
        "sources": SOURCES,
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(
            (root / "finitefree/dpp.py").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sympy": sp.__version__,
        "flint": flint.__version__,
        "machine": platform.machine(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor(),
        "selection": "A separate 20-matrix timing pilot preceded choosing the fixed 2000-per-size default ceiling; pilot draws are excluded and outcomes never determine stopping.",
        "cpu_visible_threads": os.cpu_count(),
        "thread_settings": {
            k: os.environ.get(k)
            for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "precision": "float64 numerical quadrature and eigensolves; rational expected polynomials/norms",
        "interpretation": "Theoretical count identities and Bernstein inequality apply to the exact restricted operator. All plotted theoretical values use numerical quadrature; convergence diagnostics are not rigorous error enclosures. Tail confidence intervals are pointwise, not simultaneous. Zero observed exceedances does not establish zero probability. Each matrix is one independent observation; matrix eigenvalues are not independent. n changes the spectral statistic distribution; R controls only Monte Carlo estimation precision. Expected-polynomial roots are not the centering measure.",
        "records": records,
    }


def plots(report: dict[str, Any], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    records = report["records"]
    colors = ["#126782", "#b65d14", "#6c4796"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for k, row in enumerate(records):
        n = row["dimension"]
        theory = row["theory"]
        support = np.arange(n + 1) / n - theory["mean_fraction"]
        empirical = (
            np.bincount(row["observed_counts"], minlength=n + 1) / row["samples"]
        )
        keep = (np.asarray(theory["pmf"]) > 1e-5) | (empirical > 0)
        axes[0].plot(
            support[keep],
            np.asarray(theory["pmf"])[keep],
            "-o",
            color=colors[k % 3],
            ms=3,
            label=f"n={n} count law",
        )
        axes[0].scatter(
            support[keep],
            empirical[keep],
            facecolors="none",
            edgecolors=colors[k % 3],
            s=36,
        )
        axes[1].errorbar(
            n,
            row["sample_variance_about_theory_mean"],
            yerr=1.96 * row["variance_standard_error"],
            fmt="o",
            color=colors[k % 3],
            capsize=4,
        )
    axes[0].set(
        xlabel="Interval fraction minus finite-size mean",
        ylabel="Probability mass",
        title="Fractions narrow as matrix size n grows",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(
        [r["dimension"] for r in records],
        [r["theory"]["variance_fraction"] for r in records],
        color="#222222",
        label="Numerical finite-size variance",
    )
    axes[1].set(
        xlabel="Matrix size n",
        ylabel="Variance of interval fraction",
        yscale="log",
        title="Variance and Monte Carlo uncertainty",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"GUE counts in [-0.5, 0.5] · R={report['samples_per_dimension']:,} independent matrices per n",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Left: solid = numerical count law; hollow = Monte Carlo. Right: bars = ±1.96 estimated Monte Carlo SE. Quadrature is not certified.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(str(prefix) + "-variance.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(records), figsize=(4 * len(records), 4.4), squeeze=False
    )
    for ax, row in zip(axes[0], records):
        tails = row["tails"]
        eps = np.array([t["epsilon"] for t in tails])
        empirical = np.array([t["empirical_probability"] for t in tails])
        lower = np.array([t["pointwise_cp95"][0] for t in tails])
        upper = np.array([t["pointwise_cp95"][1] for t in tails])
        nonzero = empirical > 0
        # Display zero observations as one-sided upper limits, never as log(0).
        ax.errorbar(
            eps[nonzero],
            empirical[nonzero],
            yerr=[
                empirical[nonzero] - lower[nonzero],
                upper[nonzero] - empirical[nonzero],
            ],
            fmt="o",
            ms=3,
            color="#126782",
            capsize=2,
            label="MC + pointwise CP 95%",
        )
        zero_upper = np.array(
            [t["zero_hit_one_sided_95_upper"] or np.nan for t in tails]
        )
        ax.scatter(
            eps[~nonzero],
            zero_upper[~nonzero],
            marker="v",
            s=24,
            color="#126782",
            label="0 hits: one-sided 95% upper",
        )
        # Plot the actual discrete jumps rather than interpolating sampled
        # thresholds as if the interval fraction had a continuous law.
        deviation_support = np.abs(
            np.arange(row["dimension"] + 1) / row["dimension"]
            - row["theory"]["mean_fraction"]
        )
        jumps = deviation_support[deviation_support <= eps[-1]]
        curve_x = np.unique(
            np.concatenate(([0.0, eps[-1]], jumps, np.nextafter(jumps, np.inf)))
        )
        pmf = np.asarray(row["theory"]["pmf"])
        curve_y = [
            min(1.0, float(np.sum(pmf[deviation_support >= x]))) for x in curve_x
        ]
        ax.plot(curve_x, curve_y, color="#222222", label="Numerical count-law tail")
        ax.plot(
            eps,
            [t["numerical_bernstein_reference"] for t in tails],
            "--",
            color="#b65d14",
            label="Numerical Bernstein reference",
        )
        ax.set(
            yscale="log",
            ylim=(1e-5, 1.5),
            xlabel="Deviation ε of fraction from mean",
            title=f"n={row['dimension']}",
        )
        ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel("P(|fraction − mean| ≥ ε)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"GUE interval-fraction tails · R={report['samples_per_dimension']:,} per n",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.015,
        "Triangles are upper limits, not measured probabilities. Intervals are pointwise. Numerical theory is not a certified bound.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(str(prefix) + "-tails.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31841)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-prefix", type=Path)
    args = parser.parse_args()
    report = run_showcase(
        args.dimensions, args.samples, args.seed, args.quadrature_order
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.plot_prefix:
        plots(report, args.plot_prefix)
    for row in report["records"]:
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "dimension",
                        "samples",
                        "sample_seconds",
                        "sample_mean_fraction",
                        "mean_disagreement",
                        "sample_variance_about_theory_mean",
                        "variance_disagreement",
                    ]
                }
            )
        )


if __name__ == "__main__":
    main()
