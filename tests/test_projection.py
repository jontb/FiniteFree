from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    FiniteRTransform,
    RealRootedPolynomial,
    gue_expected_poly,
    hermite_polynomial,
)


@pytest.mark.parametrize("dimension", range(6))
@pytest.mark.parametrize("leading", [1, flint.fmpq(7, 3)])
def test_projection_matches_independent_symbolic_derivative(
    dimension: int, leading: Any
) -> None:
    source = RealRootedPolynomial.from_roots([sp.Rational(-2, 3), 0, 1, 2, 4])
    p = RealRootedPolynomial(source._fmpq_poly * leading, monic=False)
    x = sp.Symbol("x")
    symbolic = sp.Poly.from_list(list(p.coeffs), x)
    projected = p.projection(dimension)
    if dimension == p.degree:
        assert projected is p
        expected = symbolic
    else:
        expected = symbolic.diff((x, p.degree - dimension)).monic()
    assert list(projected.coeffs) == expected.all_coeffs()


def test_projection_composes_and_preserves_normalized_prefix() -> None:
    p = RealRootedPolynomial.from_roots([-3, -1, 0, 0, 1, 2, 4, 5])
    cached = list(p.normalized_coeffs())
    for larger, smaller in [(7, 4), (6, 2), (4, 0)]:
        direct = p.projection(smaller)
        composed = p.projection(larger).projection(smaller)
        assert list(direct.coeffs) == list(composed.coeffs)
        assert list(direct.normalized_coeffs()) == cached[: smaller + 1]
    assert list(p.normalized_coeffs()) == cached


def test_projection_preserves_finite_cumulant_scaling_law() -> None:
    p = RealRootedPolynomial.from_roots([-3, -1, 0, 0, 1, 2, 4, 5])
    dimension = 4
    source = FiniteRTransform(p, order=dimension)
    projected = FiniteRTransform(p.projection(dimension), order=dimension)
    expected = [
        value * sp.Rational(dimension, p.degree) ** (n - 1)
        for n, value in enumerate(source, start=1)
    ]
    assert projected == expected


def test_projection_commutes_with_exact_affine_transforms() -> None:
    p = RealRootedPolynomial.from_roots([-2, -1, 0, 2, 5, 6])
    transformed = p.shift(sp.Rational(7, 3)).dilation(sp.Rational(-2, 5))
    expected = p.projection(3).shift(sp.Rational(7, 3)).dilation(sp.Rational(-2, 5))
    assert list(transformed.projection(3).coeffs) == list(expected.coeffs)


def test_gaussian_projection_retains_proven_recurrence_and_exact_family() -> None:
    source = (
        gue_expected_poly(100).shift(sp.Rational(7, 3)).dilation(sp.Rational(-2, 5))
    )
    projected = source.projection(36)
    expected = (
        hermite_polynomial(36, physicist=False)
        .dilation(sp.Rational(1, 25))
        .shift(sp.Rational(-14, 15))
    )
    assert list(projected.coeffs) == list(expected.coeffs)
    assert projected._root_recurrence is not None
    assert projected._hermite_variance == source._hermite_variance
    assert projected._hermite_center == source._hermite_center
    np.testing.assert_allclose(
        projected.evaluate_roots_float64(exact=False),
        expected.evaluate_roots_float64(exact=True),
        rtol=2e-13,
        atol=2e-13,
    )


def test_projection_does_not_infer_family_or_promote_unverified_geometry() -> None:
    generic = RealRootedPolynomial(gue_expected_poly(12).coeffs)
    projected = generic.projection(6)
    assert projected._hermite_variance is None
    assert projected._root_recurrence is None
    assert not projected._is_verified
    formal = RealRootedPolynomial([1, 0, 10, 0, 1]).projection(2)
    assert not formal._is_verified
    with pytest.raises(ValueError, match="not real-rooted"):
        formal.verify_real_rootedness()


def test_projection_keeps_representable_roots_when_variance_underflows() -> None:
    p = gue_expected_poly(100).dilation(sp.Rational(1, 10**200))
    projected = p.projection(2)
    assert projected._root_recurrence is not None
    assert p._root_recurrence is not None
    assert float(projected._hermite_variance) == 0
    assert not np.shares_memory(projected._root_recurrence[0], p._root_recurrence[0])
    np.testing.assert_allclose(
        projected.evaluate_roots_float64(exact=False),
        [-1e-201, 1e-201],
        rtol=2e-15,
        atol=0,
    )


def test_projection_avoids_intermediate_derivatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = gue_expected_poly(600)

    def unexpected_derivative(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("constructed an intermediate derivative")

    monkeypatch.setattr(p, "derivative", unexpected_derivative)
    assert p.projection(20).degree == 20
    assert list(p.projection(0).coeffs) == [1]


@pytest.mark.parametrize("dimension", [-1, 5, 2.5, None])
def test_projection_requires_an_integer_in_range(dimension: Any) -> None:
    p = RealRootedPolynomial.from_roots([1, 2, 3, 4])
    with pytest.raises(ValueError, match="integer between 0 and degree"):
        p.projection(dimension)


def test_projection_accepts_numpy_integer_and_preserves_identity_boundary() -> None:
    p = RealRootedPolynomial.from_roots([1, 2, 3, 4])
    assert p.projection(np.int64(2)).degree == 2
    assert p.projection(p.degree) is p
    constant = RealRootedPolynomial([3], monic=False)
    assert constant.projection(0) is constant
