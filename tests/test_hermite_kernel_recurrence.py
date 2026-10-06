import math
from typing import Any

import numpy as np
import pytest
import sympy as sp

from finitefree import (
    OrthogonalPolynomialKernel,
    RealRootedPolynomial,
    hermite_polynomial,
)


def kernel(n: int) -> OrthogonalPolynomialKernel:
    return OrthogonalPolynomialKernel(
        [hermite_polynomial(j, physicist=False) for j in range(n + 1)],
        [math.factorial(j) for j in range(n)],
    )


@pytest.mark.parametrize("n", [1, 2, 20, 40, 80])
@pytest.mark.parametrize(
    "points", [(2.0, 2.000000001), (0.5, 0.5), (-2.0, -1.999999999), (0.3, 1.2)]
)
def test_verified_hermite_sum_against_independent_exact_basis(
    n: int, points: tuple[float, float]
) -> None:
    x, y = points
    symbol = sp.Symbol("z")
    reference = sum(
        sp.Poly(sp.hermite_prob(j, symbol), symbol).eval(sp.Rational(x))
        * sp.Poly(sp.hermite_prob(j, symbol), symbol).eval(sp.Rational(y))
        / math.factorial(j)
        for j in range(n)
    )
    result = kernel(n)
    expected = float(reference)
    assert abs(result(x, y) - expected) / max(1, abs(expected)) < 3e-14
    assert result(x, y) == result(y, x)


def test_kernel_owns_input_basis_norms_and_exports() -> None:
    polys = [hermite_polynomial(j, physicist=False) for j in range(4)]
    norms = [1, 1, 2]
    leading = [1, 1, 1, 1]
    k = OrthogonalPolynomialKernel(polys, norms, leading)
    expected = k(0.5, 0.500000001)
    polys[1]._fmpq_poly[0] = 100
    polys.clear()
    norms[:] = [99]
    leading[:] = [99]
    exported = k.polys
    exported[1]._fmpq_poly[0] = 200
    exported.clear()
    k.norms.clear()
    k.leading_coeffs.clear()
    assert k(0.5, 0.500000001) == expected
    assert k(2, 3) == 1 + 2 * 3 + sp.Rational((2**2 - 1) * (3**2 - 1), 2)


def test_recurrence_avoids_monomial_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    k = kernel(80)

    def unexpected(*args: Any) -> Any:
        pytest.fail("Verified Hermite kernel evaluated monomial coefficients")

    monkeypatch.setattr(RealRootedPolynomial, "evaluate", unexpected)
    assert math.isfinite(k(2.0, 2.000000001))
    assert math.isfinite(k(0.5, 0.5))


def test_wrong_norms_do_not_activate_family_recurrence() -> None:
    polys = [hermite_polynomial(j, physicist=False) for j in range(3)]
    k = OrthogonalPolynomialKernel(polys, [2, 2])
    assert k(0.5, 0.500000001) == pytest.approx((1 + 0.5 * 0.500000001) / 2)


def test_changed_basis_is_not_classified_by_provenance() -> None:
    polys = [hermite_polynomial(j, physicist=False) for j in range(4)]
    polys[1]._fmpq_poly[0] = 1
    k = OrthogonalPolynomialKernel(polys, [1, 1, 2])
    x, y = 0.5, 0.500000001
    expected = 1 + (x + 1) * (y + 1) + (x * x - 1) * (y * y - 1) / 2
    assert k(x, y) == pytest.approx(expected)


@pytest.mark.parametrize("points", [(np.nan, 0.0), (np.inf, 0.0), (0.0, -np.inf)])
def test_numerical_coordinates_must_be_finite(points: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        kernel(4)(*points)


def test_numerical_range_error_retains_exact_evaluation() -> None:
    k = kernel(2)
    with pytest.raises(RuntimeError, match="float64"):
        k(1e200, 1e200)
    assert k(10**200, 10**200) == 1 + 10**400


def test_zero_basis_does_not_allocate_recurrence() -> None:
    assert kernel(0)(0.5, 0.5) == 0
