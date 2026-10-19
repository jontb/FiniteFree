import itertools
import math
from typing import Any

import flint
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
from tests._cases import case


def test_normalized_coefficients_match_root_definition() -> None:
    leading: Any
    dimension: int
    for dimension in [4, 6]:
        for leading in [
            2,
            -3,
            flint.fmpq(7, 3),
            flint.fmpq(10**400),
            flint.fmpq(1, 10**400),
        ]:
            with case(leading=leading, dimension=dimension):
                roots = [sp.Rational(1, 3), 1, 2, 4]
                base = RealRootedPolynomial.from_roots(roots)
                p = RealRootedPolynomial(base._fmpq_poly * leading, monic=False)
                stored = list(p.coeffs)
                padded = roots + [0] * (dimension - len(roots))
                expected = [
                    sum(
                        (
                            math.prod(group)
                            for group in itertools.combinations(padded, k)
                        ),
                        sp.Integer(0),
                    )
                    / math.comb(dimension, k)
                    for k in range(dimension + 1)
                ]
                assert list(p.normalized_coeffs(dimension)) == expected
                assert [
                    sp.Rational(int(v.p), int(v.q))
                    for v in p._normalized_coeffs_flint(dimension)
                ] == expected
                assert list(p.coeffs) == stored
                reconstructed = RealRootedPolynomial.from_normalized_coeffs(expected)
                assert reconstructed.degree == dimension


def test_convolutions_normalize_nonmonic_representations() -> None:
    convolution: Any
    expected: list[Any]
    for convolution, expected in [
        (symmetric_additive, [1, -6, sp.Rational(17, 2)]),
        (multiplicative, [1, sp.Rational(-9, 2), 4]),
        (asymmetric_additive, [1, -6, sp.Rational(25, 4)]),
    ]:
        with case(convolution=convolution, expected=expected):
            p = RealRootedPolynomial([2, -6, 4], monic=False)
            q = RealRootedPolynomial([-3, 9, -6], monic=False)
            assert list(convolution(p, q, 2).coeffs) == expected


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
