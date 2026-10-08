"""Independent normalization and numerical checks for the exported dashboards."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sympy as sp
from scipy.special import roots_genlaguerre

from finitefree import FiniteTTransform, OrthogonalPolynomialKernel, hermite_polynomial
from finitefree.ensembles import wishart_expected_poly

MODULE = Path(__file__).resolve().parents[2] / "examples" / "spectral_dashboards.py"
spec = importlib.util.spec_from_file_location("spectral_dashboards", MODULE)
assert spec is not None and spec.loader is not None
dashboards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboards)


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
