from fractions import Fraction
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    FiniteRTransform,
    FiniteSTransform,
    FiniteTTransform,
    RealRootedPolynomial,
    gue_expected_poly,
    wishart_expected_poly,
)


@pytest.mark.parametrize("degree", [8, 60, 150])
@pytest.mark.parametrize("shift", [10**50, -(10**50)])
def test_numerical_cumulants_preserve_shifted_semicircle_contract(
    degree: int, shift: int
) -> None:
    # Scaled Hermite has finite cumulants (0, 1, 0, ...), at every degree.
    p = gue_expected_poly(degree).shift(shift)
    expected = [float(shift), 1.0] + [0.0] * 10
    actual = FiniteRTransform(p, order=12, numerical=True, prec=128)
    assert actual[0] == expected[0]
    np.testing.assert_allclose(actual[1:], expected[1:], rtol=0, atol=1e-28)


@pytest.mark.parametrize("degree", [20, 60, 150, 300])
def test_numerical_cumulants_preserve_shifted_free_poisson_contract(
    degree: int,
) -> None:
    # With n=2d, the finite cumulants are exactly (1, 1/2, 1/4, ...).
    p = wishart_expected_poly(degree, 2 * degree).shift(10**50)
    expected = [1e50] + [2.0**-k for k in range(1, 12)]
    actual = FiniteRTransform(p, order=12, numerical=True, prec=128)
    assert actual[0] == expected[0]
    np.testing.assert_allclose(actual[1:], expected[1:], rtol=1e-14, atol=0)


def test_centering_uses_ambient_dimension_and_preserves_cached_coefficients() -> None:
    p = RealRootedPolynomial.from_roots([sp.Rational(-2, 3), 1, 4])
    cached = list(p._normalized_coeffs_flint())
    for dimension in [3, 5]:
        exact = FiniteRTransform(p, order=7, d=dimension)
        numeric = FiniteRTransform(p, order=7, d=dimension, numerical=True, prec=128)
        np.testing.assert_allclose(numeric, [float(v) for v in exact], rtol=1e-14)
    assert list(p._normalized_coeffs_flint()) == cached


def test_numerical_cumulants_retain_tiny_spread_beneath_large_shift() -> None:
    p = RealRootedPolynomial.from_roots([sp.Rational(-2, 3), 1, 4]).dilation(
        sp.Rational(1, 10**30)
    )
    expected = FiniteRTransform(p, order=3)
    actual = FiniteRTransform(p.shift(10**50), order=3, numerical=True, prec=128)
    np.testing.assert_allclose(
        actual[1:], [float(v) for v in expected[1:]], rtol=1e-14, atol=0
    )


def test_numerical_cumulants_restore_precision_after_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = RealRootedPolynomial.from_roots([1, 2, 3])
    old_precision = flint.ctx.prec

    def fail_conversion(value: Any) -> Any:
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(flint, "arb", fail_conversion)
    with pytest.raises(RuntimeError, match="conversion failed"):
        FiniteRTransform(p, numerical=True, prec=128)
    assert flint.ctx.prec == old_precision


@pytest.mark.parametrize(
    "t",
    [sp.Rational(1, 10**400), Fraction(1, 10**400), flint.fmpq(1, 10**400)],
)
def test_t_transform_keeps_valid_rational_values_near_zero(t: Any) -> None:
    p = RealRootedPolynomial.from_roots([1, 2, 3, 4, 5])
    assert FiniteTTransform(p)(t) == sp.Rational(300, 137)


@pytest.mark.parametrize(
    "t",
    [1 - sp.Rational(1, 10**20), Fraction(10**20 - 1, 10**20)],
)
def test_t_transform_keeps_valid_rational_values_near_one(t: Any) -> None:
    p = RealRootedPolynomial.from_roots([1, 2, 3, 4, 5])
    assert FiniteTTransform(p)(t) == 3


def test_t_transform_uses_right_continuous_exact_grid_and_stored_float_values() -> None:
    p = RealRootedPolynomial.from_roots([1, 2, 3, 4, 5])
    transform = FiniteTTransform(p)
    s = FiniteSTransform(p)
    for k in range(1, p.degree):
        assert transform(sp.Rational(p.degree - k, p.degree)) == 1 / s[k - 1]
    # float(3/5) lies below 3/5, but multiplication by 5 rounds up to 3.
    assert Fraction.from_float(0.6) < Fraction(3, 5)
    assert transform(0.6) == sp.Rational(45, 17)
    assert transform(np.float64(0.6)) == sp.Rational(45, 17)
    assert transform(sp.Rational(3, 5)) == sp.Rational(17, 6)
    assert transform(np.nextafter(0.6, 1)) == sp.Rational(17, 6)
    assert transform(sp.sqrt(2) / 2) == sp.Rational(17, 6)


def test_t_transform_preserves_zero_atom_boundary() -> None:
    transform = FiniteTTransform(RealRootedPolynomial.from_roots([0, 0, 0, 1, 2]))
    assert transform(0.6) == 0
    assert transform(sp.Rational(3, 5) - sp.Rational(1, 10**400)) == 0
    assert transform(sp.Rational(3, 5)) == sp.Rational(1, 3)


@pytest.mark.parametrize(
    "t", [np.float32("0.6"), np.float64("0.6"), np.longdouble("0.6000000000000000001")]
)
def test_t_transform_preserves_numpy_float_storage_precision(t: Any) -> None:
    transform = FiniteTTransform(RealRootedPolynomial.from_roots([1, 2, 3, 4, 5]))
    value = Fraction(*t.as_integer_ratio())
    # longdouble can be float64 on some platforms, so use the value actually
    # represented by that platform rather than assuming its mantissa width.
    expected = sp.Rational(45, 17) if value < Fraction(3, 5) else sp.Rational(17, 6)
    assert transform(t) == expected


@pytest.mark.parametrize("degree", [20, 60, 150])
def test_t_transform_matches_marchenko_pastur_diagonal_limit(degree: int) -> None:
    # For n=2d, the limiting T(t) is 1/2+t/2. At a grid point the
    # finite scaled-Laguerre ratio has the explicit correction 1/n.
    n = 2 * degree
    t = sp.Rational(3, 5)
    transform = FiniteTTransform(wishart_expected_poly(degree, n))
    limit = sp.Rational(1, 2) + t / 2
    assert transform(t) - limit == sp.Rational(1, n)


@pytest.mark.parametrize(
    "t", [0, 1, -1, float("nan"), float("inf"), 1j, sp.Symbol("t"), None]
)
def test_t_transform_rejects_invalid_domain_values(t: Any) -> None:
    p = RealRootedPolynomial.from_roots([1, 2])
    with pytest.raises(ValueError, match="open interval"):
        FiniteTTransform(p)(t)


def test_t_transform_rejects_degree_zero() -> None:
    with pytest.raises(ValueError, match="positive degree"):
        FiniteTTransform(RealRootedPolynomial([1]))
