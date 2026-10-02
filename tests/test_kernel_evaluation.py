import math
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    OrthogonalPolynomialKernel,
    gap_probability_discrete,
    hermite_polynomial,
    legendre_polynomial,
)


def hermite_kernel(n: int) -> OrthogonalPolynomialKernel:
    return OrthogonalPolynomialKernel(
        [hermite_polynomial(j, physicist=False) for j in range(n + 1)],
        [math.factorial(j) for j in range(n)],
    )


@pytest.mark.parametrize("representation", [int, np.int64, sp.Integer, flint.fmpq])
def test_exact_distinct_large_integers_are_not_diagonal(representation: Any) -> None:
    kernel = hermite_kernel(2)
    x = representation(10**16)
    y = representation(10**16 + 1)
    assert kernel(x, y) == 1 + (10**16) * (10**16 + 1)
    assert kernel(y, x) == kernel(x, y)


def test_exact_inputs_do_not_require_float64_range() -> None:
    kernel = hermite_kernel(2)
    x, y = 10**400, 10**400 + 1
    assert kernel(x, y) == 1 + x * y
    assert kernel(x, x) == 1 + x * x


@pytest.mark.parametrize("x", [0.5, 2.0, 20.0, 1e6])
@pytest.mark.parametrize("separation", [1e-9, 1e-8])
def test_nearby_float_points_preserve_the_off_diagonal_kernel(
    x: float, separation: float
) -> None:
    kernel = hermite_kernel(2)
    y = x + separation
    expected = float(1 + sp.Rational(x) * sp.Rational(y))
    assert kernel(x, y) == pytest.approx(expected, rel=2e-15)
    assert kernel(x, y) == kernel(y, x)


@pytest.mark.parametrize("n", [5, 20, 40, 60])
@pytest.mark.parametrize("x", [0.5, 2.0])
def test_nearby_hermite_kernel_matches_exact_basis_reference(n: int, x: float) -> None:
    kernel = hermite_kernel(n)
    y = float(np.nextafter(x, np.inf))
    x_exact, y_exact = sp.Rational(x), sp.Rational(y)
    reference = sum(
        p.evaluate(x_exact) * p.evaluate(y_exact) / math.factorial(j)
        for j, p in enumerate(kernel.polys[:-1])
    )
    assert kernel(x, y) == pytest.approx(float(reference), rel=2e-12)
    assert kernel(x, y) == kernel(y, x)


def test_nearby_legendre_kernel_preserves_basis_normalization() -> None:
    n = 20
    polys = [legendre_polynomial(j) for j in range(n + 1)]
    # Account for the factory's normalization using its leading coefficient
    # relative to the traditional Legendre polynomial.
    scales = [
        sp.Rational(p.coeffs[0]) / sp.LC(sp.legendre(j, sp.Symbol("x")), sp.Symbol("x"))
        for j, p in enumerate(polys)
    ]
    norms = [2 * scales[j] ** 2 / (2 * j + 1) for j in range(n)]
    kernel = OrthogonalPolynomialKernel(polys, norms)
    x, y = 0.3, float(np.nextafter(0.3, np.inf))
    x_exact, y_exact = sp.Rational(x), sp.Rational(y)
    reference = sum(
        p.evaluate(x_exact) * p.evaluate(y_exact) / norm
        for p, norm in zip(polys, norms)
    )
    assert kernel(x, y) == pytest.approx(float(reference), rel=2e-12)


def test_diagonal_and_separated_points_keep_christoffel_darboux_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = hermite_kernel(5)

    def unexpected_basis_evaluation(value: Any) -> Any:
        raise AssertionError("unnecessary lower-basis evaluation")

    monkeypatch.setattr(kernel.polys[0], "evaluate", unexpected_basis_evaluation)
    assert math.isfinite(kernel(0.5, 0.5))
    assert math.isfinite(kernel(0.5, 1.2))


def test_empty_basis_kernel_and_gap_are_identity() -> None:
    kernel = hermite_kernel(0)
    assert kernel(1, 2) == 0
    assert kernel(1.0, 1.0) == 0
    assert kernel.k_point_correlation([0, 1]) == 0
    assert gap_probability_discrete(kernel, [0, 1]) == 1
