"""Compose GUE sampling, expected polynomials, DPP density and gap comparison.

Run with PYTHONPATH=. python examples/gue_spectral_workflow.py --output /tmp/gue.json
Add --plot /tmp/gue.png for a figure (requires optional Matplotlib).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.special import roots_hermitenorm

from finitefree import (
    EmpiricalComparison,
    OrthogonalPolynomialKernel,
    gap_probability_continuous,
    gue_expected_poly,
    sample_gue,
)


def run_workflow(
    dimension: int = 8, samples: int = 256, seed: int = 1729
) -> dict[str, Any]:
    """Return distinct finite-size spectral measures on one consistent scale."""
    if dimension < 1 or samples < 2:
        raise ValueError("dimension must be positive and samples at least two")
    p = gue_expected_poly(dimension)
    # All degrees use the SAME variance 1/d and normalized Gaussian measure.
    basis = [p.projection(j) for j in range(dimension + 1)]
    norms = [sp.factorial(j) / sp.Integer(dimension) ** j for j in range(dimension)]
    kernel = OrthogonalPolynomialKernel(basis, norms)

    def weight(x: Any) -> Any:
        return np.sqrt(dimension / (2 * np.pi)) * np.exp(-dimension * x * x / 2)

    # The existing matrix sampler uses NumPy's global RNG; restore caller state.
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        comparison = EmpiricalComparison(p, samples, lambda: sample_gue(dimension))
    finally:
        np.random.set_state(rng_state)
    roots = np.asarray(p.evaluate_roots_float64(), dtype=float)
    grid = np.linspace(-3.0, 3.0, 501)
    density = np.array([weight(x) * kernel(x, x) / dimension for x in grid])
    # Gaussian quadrature integrates these polynomial moments exactly in exact
    # arithmetic. This numerical check does not truncate the Gaussian tails.
    nodes, weights = roots_hermitenorm(dimension + 1)
    nodes /= math.sqrt(dimension)
    weights /= math.sqrt(2 * np.pi)
    diagonal = np.array([kernel(x, x) / dimension for x in nodes])
    moments = [float(np.sum(weights * diagonal * nodes**k)) for k in range(3)]
    per_matrix_second = np.mean(comparison.eigenvalues**2, axis=1)
    a, b = -0.25, 0.25
    gaps = ~np.any(
        (comparison.eigenvalues >= a) & (comparison.eigenvalues <= b), axis=1
    )
    gap_frequency = float(np.mean(gaps))
    # Wilson interval quantifies Monte Carlo uncertainty, even with zero hits.
    z = 1.959963984540054
    denominator = 1 + z * z / samples
    center = (gap_frequency + z * z / (2 * samples)) / denominator
    radius = (
        z
        * np.sqrt(
            gap_frequency * (1 - gap_frequency) / samples + z * z / (4 * samples**2)
        )
        / denominator
    )
    gap_40 = gap_probability_continuous(kernel, a, b, n_points=40, weight_func=weight)
    gap_80 = gap_probability_continuous(kernel, a, b, n_points=80, weight_func=weight)
    return {
        "dimension": dimension,
        "samples": samples,
        "seed": seed,
        "measure": "normalized Gaussian: sqrt(d/(2*pi))*exp(-d*x^2/2) dx",
        "mean_esd_density_grid": grid.tolist(),
        "mean_esd_density": density.tolist(),
        "expected_polynomial_roots": roots.tolist(),
        "sampled_eigenvalues": comparison.eigenvalues.tolist(),
        "mean_esd_moments_mass_mean_second": moments,
        "expected_polynomial_root_second_moment": float(np.mean(roots**2)),
        "exact_mean_esd_second_moment": 1,
        "exact_polynomial_root_second_moment": str(
            sp.Rational(dimension - 1, dimension)
        ),
        "sample_mean_second_moment": float(np.mean(per_matrix_second)),
        "sample_second_moment_standard_error": float(
            np.std(per_matrix_second, ddof=1) / np.sqrt(samples)
        ),
        "expected_characteristic_coefficients": [float(c) for c in p.coeffs],
        "sample_mean_characteristic_coefficients": np.mean(
            comparison.char_poly_coeffs, axis=0
        ).tolist(),
        "coefficient_confidence_check": comparison.verify_coefficients(),
        "gap_interval": [a, b],
        "sample_gap_frequency": gap_frequency,
        "sample_gap_wilson_95_interval": [center - radius, center + radius],
        "gap_nystrom_40": gap_40,
        "gap_nystrom_80": gap_80,
        "gap_quadrature_change": abs(gap_80 - gap_40),
        "interpretation": "The mean ESD density and expected-polynomial root measure are distinct finite-size outputs. Matrix samples follow the continuous GUE law. Nyström gap values are quadrature approximations; their refinement difference is not a certified error bound. Coefficient agreement and sample comparisons are statistical diagnostics.",
    }


def plot_workflow(result: dict[str, Any], path: Path) -> None:
    """Plot the mean density, sampled ESDs and polynomial-root markers separately."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    eigs = np.asarray(result["sampled_eigenvalues"])
    ax.hist(eigs.ravel(), bins=55, density=True, alpha=0.3, label="Pooled sampled ESDs")
    ax.plot(
        result["mean_esd_density_grid"],
        result["mean_esd_density"],
        label="Mean ESD density from DPP kernel",
    )
    ax.plot(
        result["expected_polynomial_roots"],
        np.zeros(result["dimension"]),
        "|",
        markersize=14,
        label="Expected-polynomial roots",
    )
    ax.set(
        xlabel="Eigenvalue",
        ylabel="Probability density",
        title=f"GUE dimension {result['dimension']}; {result['samples']} matrix samples",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.dimension < 1 or args.samples < 2:
        parser.error("dimension must be positive and samples at least two")
    result = run_workflow(args.dimension, args.samples, args.seed)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot_workflow(result, args.plot)
    compact = {
        k: v
        for k, v in result.items()
        if k not in {"sampled_eigenvalues", "mean_esd_density_grid", "mean_esd_density"}
    }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
