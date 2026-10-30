from fractions import Fraction
from typing import Any, List

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    FiniteRTransform,
    RealRootedPolynomial,
    gue_expected_poly,
    inverse_finite_r,
)
from finitefree.convolutions import symmetric_additive
from tests._cases import case
from tests._cumulants import logarithm_reference, partition_reference

x = sp.Symbol("x")
R = sp.Rational


def expanded(roots: List[Any], d: int) -> List[Any]:
    """Independent oracle: monic coefficients of prod (x - r) with d-n zero roots."""
    padded = [sp.Rational(r) for r in roots] + [0] * (d - len(roots))
    return list(sp.Poly(sp.prod([x - r for r in padded]), x).all_coeffs())


def test_fixed_acceptance_table() -> None:
    table: list[tuple[List[Any], List[Any]]] = [
        ([2, 2], [1, -4, 3]),
        ([1, 3, 9], [1, -3, 0, 0]),
        ([R(7, 8), R(33, 16), R(-81, 32), R(-63, 64)], [1, R(-7, 2), R(3, 2), 4, -2]),
        ([1, 2, 4, 8], expanded([1, 3], 4)),
        ([3, 2], [1, -6, 8]),
        ([0, -1], [1, 0, R(1, 2)]),
    ]
    for cumulants, coefficients in table:
        with case(cumulants=cumulants):
            result = inverse_finite_r(cumulants)
            assert isinstance(result, RealRootedPolynomial)
            assert list(result.coeffs) == coefficients
            assert result.degree == len(cumulants)


def test_hermite_cumulants_reconstruct_expected_gue_polynomial() -> None:
    # Hermite polynomials have only a second finite cumulant; gue_expected_poly
    # builds its coefficients through an unrelated three-term recurrence.
    for d in [1, 2, 5, 12, 40]:
        with case(d=d):
            cumulants = [0, 1] + [0] * (d - 2) if d > 1 else [0]
            assert list(inverse_finite_r(cumulants).coeffs) == list(
                gue_expected_poly(d).coeffs
            )


def test_reconstruction_matches_independent_forward_references() -> None:
    root_lists: List[List[Any]] = [
        [R(1, 3), R(-5, 7), 2],
        [-2, -2, 0, 1],
        [0, 0, 0],
        [R(1, 2), R(1, 2), 3, 3, 10**20],
        [4],
    ]
    for roots in root_lists:
        n = len(roots)
        for dimension in [n, n + 2]:
            leading: Any
            for leading in [1, -3, flint.fmpq(7, 3)]:
                with case(roots=roots, d=dimension, leading=leading):
                    base = RealRootedPolynomial.from_roots(roots)
                    p = RealRootedPolynomial(base._fmpq_poly * leading, monic=False)
                    reference = logarithm_reference(p, dimension, dimension)
                    if dimension <= 4:
                        assert partition_reference(p, dimension, dimension) == (
                            reference
                        )
                    result = inverse_finite_r(reference, d=dimension)
                    assert list(result.coeffs) == expanded(roots, dimension)


def test_shift_dilation_and_additive_identities() -> None:
    roots = [R(-3, 2), 0, 2, 5]
    kappa = logarithm_reference(RealRootedPolynomial.from_roots(roots), 4, 4)
    for c in [R(5, 3), -4]:
        with case(operation="shift", c=c):
            shifted = [kappa[0] + c] + kappa[1:]
            assert list(inverse_finite_r(shifted).coeffs) == expanded(
                [r + c for r in roots], 4
            )
        with case(operation="dilation", c=c):
            dilated = [value * c ** (n + 1) for n, value in enumerate(kappa)]
            assert list(inverse_finite_r(dilated).coeffs) == expanded(
                [r * c for r in roots], 4
            )
    # Finite cumulants linearize the symmetric additive convolution.
    p = RealRootedPolynomial.from_roots([1, 2, 6])
    q = RealRootedPolynomial.from_roots([-1, 0, R(1, 2)])
    summed = [a + b for a, b in zip(FiniteRTransform(p, 3), FiniteRTransform(q, 3))]
    assert list(inverse_finite_r(summed).coeffs) == list(
        symmetric_additive(p, q, 3).coeffs
    )


def test_formal_result_is_not_certified_real_rooted() -> None:
    result = inverse_finite_r([0, -1])
    assert result._is_verified is False
    with pytest.raises(ValueError, match="not real-rooted"):
        result.verify_real_rootedness()


def test_rational_input_types_and_ownership() -> None:
    expected = [1, -4, 3]
    cumulants: Any
    for cumulants in [
        [2, 2],
        (np.int64(2), sp.Integer(2)),
        [Fraction(2), flint.fmpq(2)],
        np.array([2, 2]),
    ]:
        with case(cumulants=cumulants):
            assert list(inverse_finite_r(cumulants, d=2).coeffs) == expected
    source = [2, 2]
    first = inverse_finite_r(source)
    assert source == [2, 2]
    assert first is not inverse_finite_r(source)


def test_incomplete_and_invalid_inputs_fail_explicitly() -> None:
    for cumulants, dimension, message in [
        ([], None, "at least one value"),
        ([2, 2], 3, "exactly d cumulants"),
        ([2, 2, 0], 2, "exactly d cumulants"),
        ([2, 2], 0, "d must be a positive"),
        ([2, 2], True, "d must be a positive"),
        ([2, 2], 2.0, "d must be a positive"),
        ("22", None, "sequence of values"),
    ]:
        with case(cumulants=cumulants, d=dimension):
            with pytest.raises(ValueError, match=message):
                inverse_finite_r(cumulants, d=dimension)  # type: ignore[arg-type]
    for value in [0.5, np.float64(2.0), sp.Float(2.0), True, 1j, sp.sqrt(2), "x"]:
        with case(value=value):
            with pytest.raises(ValueError, match="rational cumulants"):
                inverse_finite_r([value, 2])
    with pytest.raises(ValueError, match="numerical must be a boolean"):
        inverse_finite_r([2, 2], numerical=1)  # type: ignore[arg-type]


def test_numerical_mode_matches_exact_rational_reconstruction() -> None:
    for roots in [[1, 3], [R(1, 2), -1, 2, 2], [R(-3, 7), 0, 1, 4, 4, 9]]:
        with case(roots=roots):
            d = len(roots)
            kappa = logarithm_reference(RealRootedPolynomial.from_roots(roots), d, d)
            result = inverse_finite_r([float(v) for v in kappa], numerical=True)
            assert result.dtype == np.float64 and result.shape == (d + 1,)
            np.testing.assert_allclose(
                result,
                [float(c) for c in expanded(roots, d)],
                rtol=1e-12,
                atol=1e-12,
            )
    exact_input = inverse_finite_r([2, 2], numerical=True)
    np.testing.assert_array_equal(exact_input, [1, -4, 3])
    exact_input[:] = 0
    np.testing.assert_array_equal(inverse_finite_r([2, 2], numerical=True), [1, -4, 3])


def test_numerical_mode_rejects_invalid_and_nonfinite_values() -> None:
    for value in [np.nan, np.inf, True, 1j, "2", None]:
        with case(value=value):
            with pytest.raises(ValueError, match="finite real numbers"):
                inverse_finite_r([value, 2.0], numerical=True)
    with pytest.raises(RuntimeError, match="finite float64 range"):
        inverse_finite_r([1e200, 0.0, 0.0], numerical=True)
