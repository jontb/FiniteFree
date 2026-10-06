import math
from fractions import Fraction
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import FiniteRTransform, RealRootedPolynomial, gue_expected_poly


def logarithm_reference(p: RealRootedPolynomial, order: int, d: int) -> list[Any]:
    """Independent truncated log generating function from stored coefficients."""
    z = sp.Symbol("z")
    coefficients = list(p.coeffs)
    moments = [
        (-1) ** k * sp.Rational(coefficients[k]) / coefficients[0] / math.comb(d, k)
        if k <= p.degree
        else 0
        for k in range(min(order, d) + 1)
    ]
    u = sum(moments[k] * z**k / math.factorial(k) for k in range(1, len(moments)))
    # Formal logarithm expansion, avoiding the library's cumulant recurrence.
    log = sp.Poly(0, z)
    power = sp.Poly(1, z)
    for k in range(1, min(order, d) + 1):
        power = sp.Poly(power.as_expr() * u, z)
        power = sp.Poly(
            sum(c * z ** n[0] for n, c in power.terms() if n[0] <= order), z
        )
        log += power * sp.Rational((-1) ** (k + 1), k)
    return [
        sp.cancel(n * (-d) ** (n - 1) * log.nth(n)) if n <= d else 0
        for n in range(1, order + 1)
    ]


@pytest.mark.parametrize("scale", [sp.Rational(7, 3), sp.Integer(10) ** 500])
@pytest.mark.parametrize("ambient", [0, 3])
@pytest.mark.parametrize("order", [0, 1, 3, 6, 9])
def test_prefix_cumulants_match_independent_log_definition(
    scale: sp.Expr, ambient: int, order: int
) -> None:
    roots = [Fraction(-2, 3), Fraction(1, 5), 0, 0, Fraction(7, 2), 5]
    base = RealRootedPolynomial.from_roots(roots)
    p = RealRootedPolynomial(
        base._fmpq_poly * flint.fmpq(int(scale.p), int(scale.q)), monic=False
    )
    d = p.degree + ambient
    expected = logarithm_reference(p, order, d)
    assert FiniteRTransform(p, order=order, d=d) == expected
    np.testing.assert_allclose(
        FiniteRTransform(p, order=order, d=d, numerical=True, prec=256),
        [float(v) for v in expected],
        rtol=1e-11,
        atol=1e-11,
    )
    assert p.coeffs[0] == scale


def test_prefix_cache_is_bounded_owned_and_never_claimed_complete() -> None:
    p = gue_expected_poly(128).shift(10**50)
    before = p.coeffs.copy()
    FiniteRTransform(p, order=12, numerical=True)
    assert p._normalized_coeffs_flint_cached is None
    assert p._normalized_coeffs_flint_prefix_cached is not None
    assert len(p._normalized_coeffs_flint_prefix_cached) == 13
    prefix = p._normalized_coeffs_flint_prefix(4)
    prefix[1] = flint.fmpq(123)
    FiniteRTransform(p, order=8)
    assert p._normalized_coeffs_flint_prefix_cached is not None
    assert len(p._normalized_coeffs_flint_prefix_cached) == 13
    FiniteRTransform(p, order=20)
    assert p._normalized_coeffs_flint_prefix_cached is not None
    assert len(p._normalized_coeffs_flint_prefix_cached) == 21
    assert p._normalized_coeffs_flint_cached is None
    cached = p._normalized_coeffs_flint_prefix_cached.copy()
    FiniteRTransform(p, order=25, d=150)
    assert p._normalized_coeffs_flint_prefix_cached == cached
    full = p.normalized_coeffs()
    assert len(full) == 129 and not full.flags.writeable
    assert p._normalized_coeffs_flint_prefix_cached is None
    assert full[1] == 10**50
    np.testing.assert_array_equal(p.coeffs, before)


def test_numerical_prefix_centering_translation_and_precision_restoration() -> None:
    p = gue_expected_poly(128).shift(10**50)
    old = flint.ctx.prec
    try:
        flint.ctx.prec = 67
        result = FiniteRTransform(p, order=12, numerical=True, prec=128)
        assert flint.ctx.prec == 67
        np.testing.assert_allclose(
            result, [float(10**50), 1] + [0] * 10, atol=1e-10, rtol=0
        )
    finally:
        flint.ctx.prec = old


def test_prefixes_pad_ambient_zeros_without_geometry_certification() -> None:
    p = RealRootedPolynomial([1, 0, 1])
    prefix = p._normalized_coeffs_flint_prefix(5, d=8)
    expected = [1, 0, sp.Rational(1, 28), 0, 0, 0]
    assert [sp.Rational(int(v.p), int(v.q)) for v in prefix] == expected
    assert not p._is_verified
    with pytest.raises(ValueError):
        FiniteRTransform(p, order=1, d=1)
    assert FiniteRTransform(p, order=-2) == []
    assert FiniteRTransform(RealRootedPolynomial([1]), order=3) == [0, 0, 0]
