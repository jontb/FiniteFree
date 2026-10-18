"""Independent normalization and numerical checks for the exported dashboards."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad, trapezoid
from scipy.special import airy, erf, eval_hermite, factorial, roots_genlaguerre

from finitefree import (
    FiniteTTransform,
    OrthogonalPolynomialKernel,
    PrecisionContext,
    hermite_polynomial,
)
from finitefree.ensembles import wishart_expected_poly

MODULE = Path(__file__).resolve().parents[2] / "examples" / "spectral_dashboards.py"
spec = importlib.util.spec_from_file_location("spectral_dashboards", MODULE)
assert spec is not None and spec.loader is not None
dashboards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboards)


def test_compound_roots_independent_coefficients_and_arb_reference() -> None:
    before = flint.ctx.prec
    errors = []
    law = dashboards.lognormal_measure()
    for row in dashboards.compound_data():
        d, n = row["d"], row["n"]
        coefficients = [
            (-1) ** k
            * math.comb(d, k)
            * sp.Rational(math.prod(range(n - k + 1, n + 1)), n**k) ** d
            for k in range(d + 1)
        ]
        p = dashboards.compound_polynomial(d)
        assert list(p.coeffs) == coefficients
        # Independent construction, twice the precision; no public root cache.
        exact = flint.fmpq_poly(
            [flint.fmpq(int(c.p), int(c.q)) for c in coefficients[::-1]]
        )
        with PrecisionContext(degree=d, prec=384):
            pairs: Any = exact.complex_roots()
            assert all(
                z.imag.is_zero() and z.real.rel_accuracy_bits() >= 80 for z, _ in pairs
            )
            reference = sorted(float(z.real) for z, mult in pairs for _ in range(mult))
        roots = np.array(row["roots"])
        np.testing.assert_allclose(roots, reference, rtol=1e-14, atol=0)
        assert np.mean(roots) == pytest.approx(1, abs=1e-14)
        second = d - (d - 1) * (sp.Rational(n - 1, n) ** d)
        assert np.mean(roots**2) == pytest.approx(float(second), abs=1e-13)
        cdf = np.interp(roots, law["x"], law["cdf"])
        errors.append(
            max(
                np.max(np.abs(cdf - np.arange(d) / d)),
                np.max(np.abs(cdf - np.arange(1, d + 1) / d)),
            )
        )
    assert flint.ctx.prec == before
    assert errors[-1] < errors[0]
    with pytest.raises(ValueError):
        dashboards.compound_polynomial(300)


def test_lognormal_physical_branch_moments_and_resolution() -> None:
    coarse = dashboards.lognormal_measure()
    fine = dashboards.lognormal_measure(4097)
    for model in [coarse, fine]:
        x, density = np.array(model["x"]), np.array(model["density"])
        assert np.all(density >= 0)
        assert np.all(np.diff(model["cdf"]) >= 0)
        assert model["cdf"][-1] == pytest.approx(1)
        np.testing.assert_allclose(model["moments"], [1, 1, 2, 5.5], atol=1e-5, rtol=0)
        # Independent Stieltjes reconstruction checks the inverse branch off support.
        for z in [-1 + 0.5j, 6 + 1j]:
            g = trapezoid(density / (z - x), x)
            assert g.imag < 0
            w = z * g - 1
            assert (1 + w) * np.exp(w) / w == pytest.approx(z, abs=3e-5)
    difference = np.max(
        np.abs(np.interp(coarse["x"], fine["x"], fine["cdf"]) - coarse["cdf"])
    )
    assert difference < 2e-6
    assert abs(fine["moments"][0] - 1) < abs(coarse["moments"][0] - 1)


def test_edge_cdf_scaling_independent_finite_rank_gram_and_airy() -> None:
    for s in [-4.0, 0.0, 3.0]:
        threshold = np.sqrt(2) + s / np.sqrt(2)
        # Rank one is exactly the Gaussian Hermite ground-state CDF.
        assert dashboards.edge_probability(1, s) == pytest.approx(
            (1 + erf(threshold)) / 2, abs=2e-13
        )
    d, s = 8, -2.0
    threshold = np.sqrt(2 * d) + s / (np.sqrt(2) * d ** (1 / 6))

    # Sylvester identity reduces the determinant to an independently integrated
    # d×d Gram matrix on the actual half-line, with explicit Hermite polynomials.
    def product(x: float, i: int, j: int) -> float:
        return float(
            eval_hermite(i, x)
            * eval_hermite(j, x)
            * np.exp(-x * x)
            / np.sqrt(np.pi * 2.0 ** (i + j) * factorial(i) * factorial(j))
        )

    gram = np.array(
        [
            [
                quad(product, threshold, np.inf, args=(i, j), epsabs=1e-12)[0]
                for j in range(d)
            ]
            for i in range(d)
        ]
    )
    assert dashboards.edge_probability(d, s) == pytest.approx(
        np.linalg.det(np.eye(d) - gram), abs=2e-12
    )
    # Airy integral Gram identity, independent of the quotient/diagonal formula.
    pts, weights = np.polynomial.legendre.leggauss(40)
    xs, weights = 5 * (pts + 1), 5 * weights
    kernel = np.array(
        [
            [
                quad(
                    lambda t, x=x, y=y: airy(x + t)[0] * airy(y + t)[0],
                    0,
                    np.inf,
                    epsabs=1e-13,
                )[0]
                for y in xs
            ]
            for x in xs
        ]
    )
    reference = np.linalg.det(
        np.eye(40) - np.sqrt(weights[:, None] * weights[None, :]) * kernel
    )
    assert dashboards.edge_probability(None, 0) == pytest.approx(reference, abs=2e-12)


def test_edge_probability_full_grid_refinement_and_tail_trace() -> None:
    data = dashboards.edge_distribution_data(dashboards.DEGREES)
    for d, values, diagnostic in zip(
        [None, *dashboards.DEGREES],
        [data["limit"], *data["curves"]],
        [data["reference_diagnostic"], *data["diagnostics"]],
    ):
        assert np.all(np.diff(values) >= 0)
        assert 0 < min(values) < max(values) < 1
        assert max(diagnostic.values()) < 1e-10

        def tail_diagonal(u: float, d: int | None = d) -> float:
            if d is None:
                ai, aip, _, _ = airy(u)
                return float(aip * aip - u * ai * ai)
            scale = 1 / (np.sqrt(2) * d ** (1 / 6))
            phi = dashboards.hermite_functions(
                d, np.array([np.sqrt(2 * d) + scale * u])
            )
            return float(scale * np.sum(phi * phi))

        # Numerical tail-trace estimate, not an interval-certified bound.
        assert quad(tail_diagonal, 10, np.inf, epsabs=1e-20)[0] < 1e-12
    assert np.max(np.abs(np.array(data["curves"][-1]) - data["limit"])) < np.max(
        np.abs(np.array(data["curves"][0]) - data["limit"])
    )


@pytest.mark.parametrize(
    "kind,gamma",
    [
        ("semicircle", 1),
        ("arcsine", 1),
        ("mp", 0.25),
        ("mp", 0.5),
        ("mp", 1),
        ("mp", 2),
    ],
)
def test_limit_cdf_includes_full_mass(kind: str, gamma: float) -> None:
    law = dashboards.limit_measure(kind, gamma)
    cdf = np.array(law["cdf"])
    assert np.all(np.diff(cdf) >= -1e-14)
    assert cdf[0] == pytest.approx(law["atom"])
    assert cdf[-1] == pytest.approx(1)
    if kind == "mp":
        assert law["atom"] == max(0, 1 - 1 / gamma)
        # Independent Lebesgue-Stieltjes quadrature includes atom contribution.
        x = np.array(law["x"])
        midpoints = (x[:-1] + x[1:]) / 2
        assert np.sum(midpoints * np.diff(cdf)) == pytest.approx(1, abs=2e-6)
        assert np.sum(midpoints**2 * np.diff(cdf)) == pytest.approx(1 + gamma, abs=5e-6)


@pytest.mark.parametrize("d,n", [(4, 16), (8, 16), (8, 8), (8, 4), (16, 8)])
def test_wishart_root_scaling_and_nullity(d: int, n: int) -> None:
    roots = np.array(dashboards.wishart_roots(d, n))
    r = min(d, n)
    reference = roots_genlaguerre(r, abs(n - d))[0] / n
    assert len(roots) == d
    assert np.count_nonzero(roots == 0) == max(0, d - n)
    np.testing.assert_allclose(roots[-r:], reference, atol=1e-12)
    assert np.mean(roots) == pytest.approx(1)
    # The exact coefficient identity is independent of any root recurrence.
    p = wishart_expected_poly(d, n)
    coefficients = [
        (-1) ** k
        * math.comb(d, k)
        * math.prod(range(n - k + 1, n + 1))
        * sp.Rational(1, n**k)
        for k in range(d + 1)
    ]
    assert list(p.coeffs) == coefficients


def test_recorded_samples_preserve_rng_and_dimensions() -> None:
    np.random.seed(923)
    before: Any = np.random.get_state()
    model = dashboards.ensemble_data([8, 16])
    after: Any = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    for name, rows in model["spectra"].items():
        groups = rows if name.startswith("wishart") or name == "laguerre" else [rows]
        for group in groups:
            for row in group:
                samples = row.get("samples", [row.get("roots")])
                for values in samples:
                    assert len(values) == row["d"]
                    assert np.all(np.isfinite(values))
                    assert np.all(np.diff(values) >= -1e-12)
                    if "n" in row:
                        assert np.count_nonzero(np.array(values) == 0) == max(
                            0, row["d"] - row["n"]
                        )


@pytest.mark.parametrize("distribution", ["rademacher", "uniform"])
def test_bounded_wigner_coordinate_normalization(distribution: str) -> None:
    np.random.seed(273)
    d = 256
    for family in ["goe", "gue"]:
        matrix = dashboards.sample_entries(d, d, family, distribution)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=0)
        entries = matrix[np.triu_indices(d, 1)] * np.sqrt(d)
        assert np.mean(np.abs(entries) ** 2) == pytest.approx(1, abs=0.012)
        assert abs(np.mean(entries)) < 0.015
        if distribution == "rademacher":
            np.testing.assert_allclose(np.abs(entries), 1, atol=1e-15)
        else:
            assert np.max(np.abs(entries.real)) <= np.sqrt(3)
        diagonal = np.diag(matrix) * np.sqrt(d / (2 if family == "goe" else 1))
        if distribution == "rademacher":
            np.testing.assert_allclose(np.abs(diagonal), 1)


@pytest.mark.parametrize("distribution", ["rademacher", "uniform"])
@pytest.mark.parametrize("beta", [1, 2, 4])
def test_bounded_covariance_gram_normalization(distribution: str, beta: int) -> None:
    np.random.seed(419)
    d, n = 128, 256
    matrix = dashboards.sample_entries(d, n, f"wishart{beta}", distribution)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues[0] > 0
    assert np.mean(eigenvalues) == pytest.approx(1, abs=0.015)
    assert np.mean(eigenvalues**2) == pytest.approx(1.5, abs=0.04)
    if beta == 4:
        np.testing.assert_allclose(eigenvalues[::2], eigenvalues[1::2], atol=1e-12)


@pytest.mark.parametrize(
    "gamma", [sp.Rational(1, 4), sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)]
)
def test_t_steps_and_right_continuous_jumps(gamma: Any) -> None:
    d, n = 8, int(8 / gamma)
    transform = FiniteTTransform(wishart_expected_poly(d, n))
    for k in range(d):
        expected = max(0, 1 - sp.Rational(d, n) + sp.Rational(k + 1, n))
        assert transform(sp.Rational(2 * k + 1, 2 * d)) == expected
        if k:
            assert transform(sp.Rational(k, d)) == expected
    with pytest.raises(ValueError):
        transform(0)
    with pytest.raises(ValueError):
        transform(1)


def test_clt_variance_and_symmetry_are_exact() -> None:
    model = dashboards.transform_data([8])
    for d, row in zip(model["clt_degrees"], model["cumulants"]):
        values = [sp.Rational(c) for c in row]
        assert values[1] == sp.Rational(d, d - 1)
        assert all(values[k] == 0 for k in [0, 2, 4, 6])
        assert all(v == 0 for v in values[d:])


def test_convolution_means_and_interlacing() -> None:
    for row in dashboards.interlacing_data()["rows"]:
        delta = float(sp.Rational(row["shift"]))
        for name in ["additive", "multiplicative"]:
            a, b = row[name]["p"], row[name]["q"]
            assert np.all(np.array(a) <= np.array(b) + 1e-9)
            assert np.all(np.array(b[:-1]) <= np.array(a[1:]) + 1e-9)
            assert np.mean(a) == pytest.approx(9 if name == "additive" else 18)
            assert np.mean(b) == pytest.approx(
                9 + delta if name == "additive" else (6 + delta) * 3
            )
        np.testing.assert_allclose(
            np.array(row["additive"]["q"]) - row["additive"]["p"], delta, atol=1e-9
        )


def test_hermite_recurrence_matches_public_exact_kernel() -> None:
    d = 8
    polynomials = [hermite_polynomial(k, physicist=True) for k in range(d + 1)]
    norms = [sp.Rational(math.factorial(k), 2**k) for k in range(d)]
    kernel = OrthogonalPolynomialKernel(polynomials, norms)
    xs = np.array([0.0, 0.25, 1.0, 2.0])
    phi = dashboards.hermite_functions(d, xs)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            weight = math.exp(-(x * x + y * y) / 2) / math.sqrt(math.pi)
            reference = (
                float(kernel(sp.Rational(float(x)), sp.Rational(float(y)))) * weight
            )
            assert np.dot(phi[:, i], phi[:, j]) == pytest.approx(reference, abs=1e-12)
    data = dashboards.kernel_data([8, 256])
    for region in ["bulk", "edge"]:
        row = data["rows"][region]
        errors = [
            np.max(np.abs(np.array(curve) - row["limit"])) for curve in row["curves"]
        ]
        assert errors[1] < errors[0]


def test_unitary_roots_are_unprojected_and_residuals_small() -> None:
    model = dashboards.unitary_data()
    for d, frames in zip(model["degrees"], model["rows"]):
        assert frames[0]["roots"] == [[1.0, 0.0]] * d
        for row in frames[1:]:
            roots = np.array([complex(*z) for z in row["roots"]])
            assert np.max(np.abs(np.abs(roots) - 1)) < 1e-6
            values = np.polyval(row["coefficients"], roots)
            denominator = np.polyval(np.abs(row["coefficients"]), np.abs(roots))
            assert np.max(np.abs(values) / denominator) < 1e-12
