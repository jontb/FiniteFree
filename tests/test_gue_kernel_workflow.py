import math
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp
from scipy.special import roots_hermitenorm

from finitefree import (
    OrthogonalPolynomialKernel,
    RealRootedPolynomial,
    gue_expected_poly,
)


def exact_rational(value: Any) -> sp.Rational:
    # SymPy 1.12 does not reliably coerce FLINT fmpq directly. Preserve the
    # exact numerator/denominator instead of passing through a float/string.
    return sp.Rational(int(value.p), int(value.q))


@pytest.mark.parametrize("d", [1, 2, 5, 12])
def test_composed_gue_measures(d: int) -> None:
    p = gue_expected_poly(d)
    basis = [p.projection(j) for j in range(d + 1)]
    norms = [sp.factorial(j) / sp.Integer(d) ** j for j in range(d)]
    kernel = OrthogonalPolynomialKernel(basis, norms)
    assert kernel._hermite_parameters is not None
    # Verify the shared exact recurrence independently of kernel classification.
    x = flint.fmpq_poly([0, 1])
    previous, current = flint.fmpq_poly([]), flint.fmpq_poly([1])
    for j, polynomial in enumerate(basis):
        assert polynomial._fmpq_poly == current
        previous, current = current, x * current - flint.fmpq(j, d) * previous
    nodes, weights = roots_hermitenorm(d + 1)
    nodes /= math.sqrt(d)
    weights /= math.sqrt(2 * math.pi)
    density_ratio = np.array([kernel(y, y) / d for y in nodes])
    moments = [np.dot(weights, density_ratio * nodes**k) for k in range(3)]
    np.testing.assert_allclose(moments, [1.0, 0.0, 1.0], atol=2e-14, rtol=2e-14)
    # Newton identities give the distinct exact polynomial-root second moment.
    coeffs = p._fmpq_poly
    second = (coeffs[d - 1] ** 2 - 2 * coeffs[d - 2]) / d if d > 1 else flint.fmpq(0)
    assert second == flint.fmpq(d - 1, d)
    assert 1 - second == flint.fmpq(1, d)
    np.testing.assert_allclose(
        np.mean(p.evaluate_roots_float64() ** 2), float(second), atol=1e-14
    )


@pytest.mark.parametrize(
    "variance,center,mass",
    [
        (sp.Rational(1, 7), sp.Rational(2, 3), sp.Rational(5, 4)),
        (sp.Integer(3), sp.Integer(-2), sp.Integer(2)),
    ],
)
def test_scaled_shifted_signed_basis(variance: Any, center: Any, mass: Any) -> None:
    n = 14
    x = sp.Symbol("x")
    previous, current = sp.Integer(0), sp.Integer(1)
    polys = []
    norms = []
    leading = []
    for j in range(n + 1):
        scale = sp.Rational((-1) ** j * (j + 2), j + 1)
        polys.append(
            RealRootedPolynomial(
                sp.Poly(scale * current, x).all_coeffs(),
                monic=False,
                assume_real_rooted=True,
            )
        )
        leading.append(scale)
        if j < n:
            norms.append(mass * sp.factorial(j) * variance**j * scale**2)
        previous, current = (
            current,
            sp.expand((x - center) * current - j * variance * previous),
        )
    kernel = OrthogonalPolynomialKernel(polys, norms, leading)
    assert kernel._hermite_parameters is not None
    for a, b in [(0.3, 1.2), (0.5, 0.5), (0.5, float(np.nextafter(0.5, 1.0)))]:
        aq, bq = sp.Rational(a), sp.Rational(b)
        expected = sum(
            exact_rational(poly.evaluate(aq)) * exact_rational(poly.evaluate(bq)) / norm
            for poly, norm in zip(polys[:-1], norms)
        )
        assert kernel(a, b) == pytest.approx(float(expected), rel=3e-13, abs=1e-12)
        assert kernel(a, b) == kernel(b, a)
        assert exact_rational(kernel(aq, bq)) == expected
    before = kernel(0.3, 1.2)
    polys[1]._fmpq_poly[0] = 999
    kernel.polys[2]._fmpq_poly[0] = 999
    assert kernel(0.3, 1.2) == before


def test_scaled_fast_path_avoids_coefficients(monkeypatch: pytest.MonkeyPatch) -> None:
    d = 180  # factorial norms overflow float64; exact ratios do not.
    p = gue_expected_poly(d)
    kernel = OrthogonalPolynomialKernel(
        [p.projection(j) for j in range(d + 1)],
        [sp.factorial(j) / sp.Integer(d) ** j for j in range(d)],
    )
    assert kernel._hermite_parameters is not None

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("monomial evaluation in verified scaled recurrence")

    monkeypatch.setattr(RealRootedPolynomial, "evaluate", forbidden)
    assert math.isfinite(kernel(0.1, float(np.nextafter(0.1, 1.0))))


def test_inconsistent_data_does_not_dispatch() -> None:
    d = 4
    p = gue_expected_poly(d)
    basis = [p.projection(j) for j in range(d + 1)]
    norms = [sp.factorial(j) / sp.Integer(d) ** j for j in range(d)]
    bad_norms = norms.copy()
    bad_norms[2] *= 2
    bad_leading = [1, 1, -1, 1, 1]
    bad_basis = [gue_expected_poly(j) for j in range(d + 1)]
    for kernel in (
        OrthogonalPolynomialKernel(basis, bad_norms),
        OrthogonalPolynomialKernel(basis, norms, bad_leading),
        OrthogonalPolynomialKernel(bad_basis, norms),
    ):
        assert kernel._hermite_parameters is None
    # Arbitrary mass is valid; normalization comes from norms, not provenance.
    kernel = OrthogonalPolynomialKernel(basis, [3 * h for h in norms])
    assert kernel._hermite_parameters is not None
    exact = sum(
        exact_rational(poly.evaluate(sp.Rational(1, 2))) ** 2 / (3 * h)
        for poly, h in zip(basis[:-1], norms)
    )
    assert kernel(0.5, 0.5) == pytest.approx(float(exact))


def test_range_failure_does_not_change_exact_queries() -> None:
    p = gue_expected_poly(2)
    kernel = OrthogonalPolynomialKernel(
        [p.projection(j) for j in range(3)],
        [sp.Rational(1, 10**800), sp.Rational(1, 2 * 10**800)],
    )
    with pytest.raises(RuntimeError, match="float64"):
        kernel(0.0, 0.0)
    assert kernel(0, 0) == 10**800
