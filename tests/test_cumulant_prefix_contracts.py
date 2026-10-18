from fractions import Fraction

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    FiniteRTransform,
    RealRootedPolynomial,
    gue_expected_poly,
    wishart_expected_poly,
)
from tests._cases import case
from tests._cumulants import assert_cumulants, logarithm_reference, partition_reference


def test_cumulants_against_independent_definitions_and_closed_forms() -> None:
    x = sp.Symbol("x")
    scenarios = [
        (
            "quadratic",
            RealRootedPolynomial([1, -3, 2]),
            [sp.Rational(3, 2), sp.Rational(1, 2), 0, 0],
        ),
        ("cubic", RealRootedPolynomial.from_roots([1, 2, 3]), [2, 1, 0]),
        (
            "wishart",
            wishart_expected_poly(6, 10),
            [sp.Rational(3, 5) ** k for k in range(6)],
        ),
        (
            "hermite",
            RealRootedPolynomial(
                sp.Poly(sp.hermite_prob(50, x), x).all_coeffs(), assume_real_rooted=True
            ),
            [0, 50] + [0] * 6,
        ),
        ("third cumulant", RealRootedPolynomial.from_roots([0, 0, 3]), [1, 3, 9]),
    ]
    for leading in (2, -3, flint.fmpq(1, 10**400), flint.fmpq(10**400), sp.sqrt(2)):
        scenarios.append(
            (
                str(leading),
                RealRootedPolynomial([leading, -3 * leading, 2 * leading], monic=False),
                [sp.Rational(3, 2), sp.Rational(1, 2)],
            )
        )
    for name, p, expected in scenarios:
        with case(polynomial=name):
            stored, verified = list(p.coeffs), p._is_verified
            for order in range(len(expected) + 1):
                assert_cumulants(p, order, expected=expected[:order], prec=128)
            assert (
                partition_reference(p, min(3, len(expected)), p.degree) == expected[:3]
            )
            if p.degree == 2:
                coefficients = p.normalized_coeffs()
                assert list(coefficients) == [1, sp.Rational(3, 2), 2]
                assert coefficients is p.normalized_coeffs()
                if p._is_flint:
                    exact = p._normalized_coeffs_flint()
                    assert exact is p._normalized_coeffs_flint()
                else:
                    assert not p._is_verified
            assert list(p.coeffs) == stored and p._is_verified == verified

    # Prefix truncation and ambient zero padding retain their original matrix.
    for order in (0, 1, 3, 6, 9):
        for ambient in (0, 3):
            for scale in (sp.Rational(7, 3), sp.Integer(10) ** 500):
                with case(scale=scale, ambient=ambient, order=order):
                    base = RealRootedPolynomial.from_roots(
                        [Fraction(-2, 3), Fraction(1, 5), 0, 0, Fraction(7, 2), 5]
                    )
                    p = RealRootedPolynomial(
                        base._fmpq_poly * flint.fmpq(int(scale.p), int(scale.q)),
                        monic=False,
                    )
                    assert_cumulants(
                        p, order, d=p.degree + ambient, rtol=1e-11, atol=1e-11
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


def test_affine_cumulants_against_logarithm_after_cache_reordering() -> None:
    """Mix cache orders, ambient dimensions and extreme nonmonic scales."""
    base = RealRootedPolynomial.from_roots([-3, sp.Rational(1, 7), 0, 0, 2, 5])
    for scale in [flint.fmpq(-7, 3), flint.fmpq(1, 10**500), flint.fmpq(10**500)]:
        with case(scale=scale):
            p = RealRootedPolynomial(base._fmpq_poly * scale, monic=False)
            stored = list(p.coeffs)
            # Warming one dimension/order must not contaminate another.
            for d, order in [(9, 3), (6, 6), (9, 9), (6, 1), (9, 2), (6, 6)]:
                with case(d=d, order=order):
                    expected = logarithm_reference(p, order, d)
                    assert FiniteRTransform(p, order=order, d=d) == expected
                    np.testing.assert_allclose(
                        FiniteRTransform(p, order=order, d=d, numerical=True),
                        [float(v) for v in expected],
                        rtol=2e-12,
                        atol=2e-12,
                    )
            shift, dilation = 10**50 + 1, sp.Rational(-3, 7)
            transformed = p.shift(shift).dilation(dilation)
            reference = logarithm_reference(transformed, 6, 6)
            original = logarithm_reference(p, 6, 6)
            assert reference == [
                dilation * (original[0] + shift),
                *[dilation**n * original[n - 1] for n in range(2, 7)],
            ]
            assert FiniteRTransform(transformed, order=6) == reference
            np.testing.assert_allclose(
                FiniteRTransform(transformed, order=6, numerical=True, prec=192),
                [float(v) for v in reference],
                rtol=2e-12,
                atol=2e-12,
            )
            assert list(p.coeffs) == stored
