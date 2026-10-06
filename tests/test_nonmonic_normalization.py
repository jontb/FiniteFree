import itertools
import math
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
)
from finitefree.convolutions import (
    asymmetric_additive,
    multiplicative,
    symmetric_additive,
)


@pytest.mark.parametrize(
    "leading", [2, -3, flint.fmpq(7, 3), flint.fmpq(10**400), flint.fmpq(1, 10**400)]
)
@pytest.mark.parametrize("dimension", [4, 6])
def test_normalized_coefficients_match_root_definition(
    leading: Any, dimension: int
) -> None:
    roots = [sp.Rational(1, 3), 1, 2, 4]
    base = RealRootedPolynomial.from_roots(roots)
    p = RealRootedPolynomial(base._fmpq_poly * leading, monic=False)
    stored = list(p.coeffs)
    padded = roots + [0] * (dimension - len(roots))
    expected = [
        sum(
            (math.prod(group) for group in itertools.combinations(padded, k)),
            sp.Integer(0),
        )
        / math.comb(dimension, k)
        for k in range(dimension + 1)
    ]
    assert list(p.normalized_coeffs(dimension)) == expected
    assert [
        sp.Rational(int(v.p), int(v.q)) for v in p._normalized_coeffs_flint(dimension)
    ] == expected
    assert list(p.coeffs) == stored
    reconstructed = RealRootedPolynomial.from_normalized_coeffs(expected)
    assert reconstructed.degree == dimension


@pytest.mark.parametrize(
    "leading", [2, -3, flint.fmpq(1, 10**400), flint.fmpq(10**400)]
)
def test_cumulants_and_cached_sequences_ignore_leading_scalar(leading: Any) -> None:
    p = RealRootedPolynomial([leading, -3 * leading, 2 * leading], monic=False)
    expected = [sp.Rational(3, 2), sp.Rational(1, 2)]
    assert FiniteRTransform(p, order=2) == expected
    np.testing.assert_allclose(
        FiniteRTransform(p, order=2, numerical=True), [1.5, 0.5], rtol=1e-14
    )
    coefficients = p.normalized_coeffs()
    assert coefficients is p.normalized_coeffs()
    flint_coefficients = p._normalized_coeffs_flint()
    assert flint_coefficients is p._normalized_coeffs_flint()
    assert list(coefficients) == [1, sp.Rational(3, 2), 2]


@pytest.mark.parametrize(
    "convolution, expected",
    [
        (symmetric_additive, [1, -6, sp.Rational(17, 2)]),
        (multiplicative, [1, sp.Rational(-9, 2), 4]),
        (asymmetric_additive, [1, -6, sp.Rational(25, 4)]),
    ],
)
def test_convolutions_normalize_nonmonic_representations(
    convolution: Any, expected: list[Any]
) -> None:
    p = RealRootedPolynomial([2, -6, 4], monic=False)
    q = RealRootedPolynomial([-3, 9, -6], monic=False)
    assert list(convolution(p, q, 2).coeffs) == expected


def test_additive_power_has_the_same_cumulants_for_nonmonic_input() -> None:
    p = RealRootedPolynomial([2, -6, 4], monic=False)
    assert list(p.additive_power(1).coeffs) == [1, -3, 2]
    assert list(p.additive_power(2).coeffs) == [1, -6, sp.Rational(17, 2)]
    assert FiniteRTransform(p.additive_power(2), order=2) == [3, 1]


def test_rational_and_symbolic_coefficient_backends_agree() -> None:
    p = RealRootedPolynomial([sp.sqrt(2), -3 * sp.sqrt(2), 2 * sp.sqrt(2)], monic=False)
    assert not p._is_flint
    stored = list(p.coeffs)
    assert list(p.normalized_coeffs()) == [1, sp.Rational(3, 2), 2]
    assert FiniteRTransform(p, order=2) == [sp.Rational(3, 2), sp.Rational(1, 2)]
    assert list(p.coeffs) == stored
    assert not p._is_verified


def test_nonunit_constant_and_nonmonic_derivative_use_root_normalization() -> None:
    constant = RealRootedPolynomial([7], monic=False)
    assert list(constant.normalized_coeffs()) == [1]
    assert list(constant.normalized_coeffs(3)) == [1, 0, 0, 0]
    p = RealRootedPolynomial.from_roots([1, 2, 3])
    derivative = p.derivative(monic=False)
    assert derivative.coeffs[0] == 3
    assert list(derivative.normalized_coeffs()) == list(
        p.derivative().normalized_coeffs()
    )
    assert FiniteRTransform(derivative, order=2) == FiniteRTransform(
        p.derivative(), order=2
    )


def test_existing_ratio_transforms_preserve_scalar_invariance() -> None:
    p = RealRootedPolynomial([2, -6, 4], monic=False)
    assert list(FiniteSTransform(p)) == [sp.Rational(2, 3), sp.Rational(3, 4)]
    assert FiniteTTransform(p)(sp.Rational(3, 4)) == sp.Rational(3, 2)
