from typing import Any, List

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import RealRootedPolynomial, root_moments
from tests._cases import case


def direct_moments(roots: List[Any], order: int, dimension: int) -> List[Any]:
    """Independent oracle: rational root power sums with embedded zero roots."""
    values = [sp.Rational(root) for root in roots]
    return [sp.Integer(1)] + [
        sum((value**k for value in values), sp.Integer(0)) / dimension
        for k in range(1, order + 1)
    ]


def test_fixed_acceptance_values() -> None:
    p = RealRootedPolynomial([1, -4, 3])  # (x-1)(x-3)
    assert root_moments(p, 4) == [1, 2, 5, 14, 41]
    half = sp.Rational(1, 2)
    assert root_moments(p, 4, d=4) == [1, 1, 5 * half, 7, 41 * half]


def test_root_moments_match_direct_power_sums() -> None:
    root_lists: List[List[Any]] = [
        [1, 3],
        [sp.Rational(1, 3), sp.Rational(-5, 7), 2],
        [-2, -2, 0, 1],
        [0, 0, 0],
        [4],
        [sp.Rational(-1, 2), sp.Rational(-1, 2), 3, 3, 10**30],
    ]
    leading: Any
    for roots in root_lists:
        n = len(roots)
        for leading in [1, -3, flint.fmpq(7, 3), flint.fmpq(10**400)]:
            for dimension in [n, n + 3]:
                with case(roots=roots, leading=leading, dimension=dimension):
                    base = RealRootedPolynomial.from_roots(roots)
                    p = RealRootedPolynomial(base._fmpq_poly * leading, monic=False)
                    order = n + 4  # orders above the degree are not truncated
                    expected = direct_moments(roots, order, dimension)
                    result = root_moments(p, order, d=dimension)
                    assert result == expected
                    assert all(isinstance(value, sp.Rational) for value in result)
                    assert root_moments(p, 0, d=dimension) == [1]


def test_descending_coefficient_input_matches_roots() -> None:
    # 6x^3 - 5x^2 - 2x + 1 = 6(x-1)(x-1/3)(x+1/2), written out by hand.
    p = RealRootedPolynomial([6, -5, -2, 1], monic=False)
    roots = [1, sp.Rational(1, 3), sp.Rational(-1, 2)]
    assert root_moments(p, 6) == direct_moments(roots, 6, 3)


def test_shift_and_dilation_follow_transformed_roots() -> None:
    roots = [sp.Rational(-3, 2), 0, 2, 5]
    p = RealRootedPolynomial.from_roots(roots)
    for c in [sp.Rational(5, 3), -4]:
        with case(operation="shift", c=c):
            shifted = [root + c for root in roots]
            assert root_moments(p.shift(c), 6, d=6) == direct_moments(shifted, 6, 6)
        with case(operation="dilation", c=c):
            dilated = [root * c for root in roots]
            assert root_moments(p.dilation(c), 6) == direct_moments(dilated, 6, 4)


def test_constant_polynomial_requires_dimension() -> None:
    p = RealRootedPolynomial([5])
    with pytest.raises(ValueError, match="explicit positive dimension"):
        root_moments(p, 2)
    assert root_moments(p, 3, d=2) == [1, 0, 0, 0]
    assert root_moments(p, 0, d=1) == [1]


def test_coefficients_only_without_real_root_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("root_moments must not validate or extract roots")

    monkeypatch.setattr(RealRootedPolynomial, "verify_real_rootedness", fail)
    monkeypatch.setattr(RealRootedPolynomial, "evaluate_roots_float64", fail)
    # x^2 + 1 has roots +-i: power sums 0, -2, 0, 2 are complex-root sums.
    p = RealRootedPolynomial([1, 0, 1])
    assert root_moments(p, 4) == [1, 0, -1, 0, 1]
    assert p._is_verified is False
    assert p._has_non_negative_roots_cached is None
    assert p._roots_cached is None


def test_invalid_indices_and_domains() -> None:
    p = RealRootedPolynomial.from_roots([1, 3])
    for order in [-1, 1.5, True, np.True_, "2", None]:
        with case(order=order):
            with pytest.raises(ValueError, match="order must be a nonnegative"):
                root_moments(p, order)  # type: ignore[arg-type]
    for dimension in [0, -2, 2.0, False]:
        with case(d=dimension):
            with pytest.raises(ValueError, match="d must be a positive"):
                root_moments(p, 2, d=dimension)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be less than polynomial degree"):
        root_moments(p, 2, d=1)
    symbolic = RealRootedPolynomial([1, -sp.sqrt(2)])
    with pytest.raises(ValueError, match="rational coefficients"):
        root_moments(symbolic, 2)


def test_integer_index_protocol_and_result_ownership() -> None:
    p = RealRootedPolynomial.from_roots([1, 3])
    assert root_moments(p, np.int64(3), d=np.int32(4)) == root_moments(p, 3, d=4)
    first = root_moments(p, 2)
    first[1] = sp.Integer(99)
    second = root_moments(p, 2)
    assert second == [1, 2, 5]
    assert second is not first
    assert list(p.coeffs) == [1, -4, 3]
