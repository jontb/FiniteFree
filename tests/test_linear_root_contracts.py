from typing import Any

import numpy as np
import pytest
import sympy as sp

from finitefree import RealRootedPolynomial, UnitaryPolynomial
from finitefree.orthogonal import jacobi_polynomial, laguerre_polynomial


@pytest.mark.parametrize("root", [sp.I, -sp.sqrt(2) * sp.I, sp.oo])
def test_nonreal_linear_roots_cannot_acquire_a_real_root_certificate(root: Any) -> None:
    p = RealRootedPolynomial([1, -root])
    for request in (
        p.verify_real_rootedness,
        lambda: p.evaluate_roots_float64(exact=False),
        lambda: p.evaluate_roots_float64(exact=True),
        lambda: p.has_non_negative_roots,
        lambda: p.has_strictly_positive_roots,
    ):
        with pytest.raises(ValueError, match="not real-rooted"):
            request()
        assert not p._is_verified
        assert p._roots_cached is None


def test_unknown_linear_root_realness_remains_uncertified() -> None:
    root = sp.Symbol("a")
    p = RealRootedPolynomial([1, -root])
    with pytest.raises(RuntimeError, match="linear root is real"):
        p.verify_real_rootedness()
    assert not p._is_verified


def test_real_symbolic_root_can_be_certified_without_claiming_its_sign_or_float() -> (
    None
):
    root = sp.Symbol("a", real=True)
    p = RealRootedPolynomial([1, -root])
    assert p.verify_real_rootedness()
    with pytest.raises(RuntimeError, match="nonnegative"):
        _ = p.has_non_negative_roots
    with pytest.raises(RuntimeError, match="positive"):
        _ = p.has_strictly_positive_roots
    with pytest.raises(RuntimeError, match="converted to float64"):
        p.evaluate_roots_float64()
    assert p._roots_cached is None


@pytest.mark.parametrize("positive", [True, False])
def test_symbolic_sign_assumptions_are_respected(positive: bool) -> None:
    root = sp.Symbol("a", positive=True) if positive else sp.Symbol("a", negative=True)
    p = RealRootedPolynomial([1, -root])
    assert p.verify_real_rootedness()
    assert p.has_non_negative_roots is positive
    assert p.has_strictly_positive_roots is positive


@pytest.mark.parametrize("root", [sp.sqrt(2), -sp.sqrt(2), sp.pi, 0])
@pytest.mark.parametrize("exact", [True, False])
def test_numeric_linear_roots_match_the_exact_coefficient_ratio(
    root: Any, exact: bool
) -> None:
    p = RealRootedPolynomial([1, -root])
    assert p.verify_real_rootedness()
    assert p.has_non_negative_roots == bool(root >= 0)
    assert p.has_strictly_positive_roots == bool(root > 0)
    actual = p.evaluate_roots_float64(exact=exact)
    assert actual.dtype == np.float64
    assert not actual.flags.writeable
    np.testing.assert_allclose(actual, [float(sp.N(root, 60))], rtol=1e-14, atol=0)
    assert actual is p.evaluate_roots_float64(exact=exact)


@pytest.mark.parametrize("leading", [2, -3, sp.Rational(7, 3)])
def test_nonmonic_linear_ratio_and_geometry_are_scalar_invariant(leading: Any) -> None:
    p = RealRootedPolynomial([leading, -sp.Rational(5, 7) * leading], monic=False)
    assert p.verify_real_rootedness()
    assert p.has_non_negative_roots
    assert p.has_strictly_positive_roots
    np.testing.assert_allclose(p.evaluate_roots_float64(), [5 / 7], rtol=1e-14)


@pytest.mark.parametrize("root", [sp.Rational(1, 10**250), 10**250])
def test_representable_linear_roots_preserve_extreme_scales(root: Any) -> None:
    p = RealRootedPolynomial([1, -root])
    np.testing.assert_allclose(p.evaluate_roots_float64(), [float(root)], rtol=1e-14)


def test_unrepresentable_linear_roots_do_not_enter_the_cache() -> None:
    p = RealRootedPolynomial([1, -(10**1000)])
    with pytest.raises(RuntimeError, match="finite float64 range"):
        p.evaluate_roots_float64()
    assert p._roots_cached is None


@pytest.mark.parametrize(
    "p, expected",
    [
        (laguerre_polynomial(1, sp.Rational(1, 2)), sp.Rational(3, 2)),
        (
            jacobi_polynomial(1, sp.Rational(1, 3), sp.Rational(2, 3)),
            sp.Rational(1, 9),
        ),
        (RealRootedPolynomial([1, 0, -1]).derivative(), 0),
    ],
)
def test_linear_family_and_derivative_roots_use_their_coefficients(
    p: RealRootedPolynomial, expected: Any
) -> None:
    np.testing.assert_allclose(
        p.evaluate_roots_float64(exact=False), [float(expected)], rtol=1e-14, atol=0
    )


def test_symbolic_linear_reconstruction_validates_and_extracts_roots() -> None:
    p = RealRootedPolynomial.from_normalized_coeffs([1, sp.sqrt(2)])
    assert not p._is_verified
    np.testing.assert_allclose(p.evaluate_roots_float64(), [np.sqrt(2)], rtol=1e-14)


def test_nonmonic_symbolic_common_factor_is_removed_before_root_certification() -> None:
    p = RealRootedPolynomial([sp.I, -sp.I * sp.sqrt(2)], monic=False)
    assert p.verify_real_rootedness()
    assert p.has_strictly_positive_roots
    assert p.evaluate(sp.sqrt(2)) == 0
    np.testing.assert_allclose(p.evaluate_roots_float64(), [np.sqrt(2)], rtol=1e-14)


@pytest.mark.parametrize("constant", [1, sp.I])
def test_nonzero_constants_have_empty_roots_and_vacuous_positive_geometry(
    constant: Any,
) -> None:
    p = RealRootedPolynomial([constant])
    assert p.verify_real_rootedness()
    assert p.has_non_negative_roots
    assert p.has_strictly_positive_roots
    assert p.evaluate_roots_float64().size == 0


def test_explicit_assumption_and_unitary_class_keep_their_distinct_contracts() -> None:
    assumed = RealRootedPolynomial([1, sp.I], assume_real_rooted=True)
    assert assumed.verify_real_rootedness()
    p = UnitaryPolynomial([1, sp.I])
    assert not p.verify_real_rootedness()
    np.testing.assert_allclose(p.evaluate_roots_float64(), [-1j])
