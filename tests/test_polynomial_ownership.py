from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    FiniteRTransform,
    FiniteTTransform,
    RealRootedPolynomial,
    UnitaryPolynomial,
    legendre_polynomial,
)


@pytest.mark.parametrize("monic", [True, False])
@pytest.mark.parametrize("warm_cache", [True, False])
def test_mutating_a_flint_input_cannot_change_coefficients_or_certificates(
    monic: bool, warm_cache: bool
) -> None:
    source = flint.fmpq_poly([4, -6, 2])
    p = RealRootedPolynomial(source, monic=monic)
    expected = [1, -3, 2] if monic else [2, -6, 4]
    if warm_cache:
        assert p.has_strictly_positive_roots
        p.evaluate_roots_float64()
        p.normalized_coeffs()
        p.evaluate(0.0)
    source[0] = 100
    assert list(p.coeffs) == expected
    assert p.degree == 2
    assert p.verify_real_rootedness()
    assert p.has_strictly_positive_roots
    assert p.evaluate(0) == expected[-1]
    assert p.evaluate(0.0) == expected[-1]
    np.testing.assert_allclose(p.evaluate_roots_float64(), [1, 2])
    # The changed caller input has discriminant 36 - 4*2*100 < 0.
    with pytest.raises(ValueError, match="not real-rooted"):
        RealRootedPolynomial(source).verify_real_rootedness()
    source[3] = 1
    assert p.degree == 2
    assert list(p.coeffs) == expected


@pytest.mark.parametrize("array", [False, True])
def test_coefficient_sequences_and_accessors_are_independent(array: bool) -> None:
    source: Any = [1, -3, 2]
    if array:
        source = np.array(source, dtype=object)
    p = RealRootedPolynomial(source)
    source[-1] = 100
    exposed = p.coeffs
    exposed[-1] = 100
    assert list(p.coeffs) == [1, -3, 2]
    assert p.evaluate(1) == 0
    assert FiniteRTransform(p, order=2) == [sp.Rational(3, 2), sp.Rational(1, 2)]


def test_symbolic_coefficients_are_owned_and_public_coefficients_are_copies() -> None:
    source = np.array([1, -sp.sqrt(2)], dtype=object)
    p = RealRootedPolynomial(source)
    source[-1] = sp.I
    exposed = p.coeffs
    exposed[-1] = sp.I
    assert p.evaluate(sp.sqrt(2)) == 0
    assert list(p.coeffs) == [1, -sp.sqrt(2)]
    with pytest.raises(ValueError, match="read-only"):
        p.coeffs_sympy[-1] = sp.I


@pytest.mark.parametrize("constructor", ["roots", "normalized"])
def test_alternative_constructors_copy_mutable_sequences(constructor: str) -> None:
    if constructor == "roots":
        source = np.array([1, 2], dtype=object)
        p = RealRootedPolynomial.from_roots(source)
    else:
        source = np.array([1, sp.Rational(3, 2), 2], dtype=object)
        p = RealRootedPolynomial.from_normalized_coeffs(source)
    source[-1] = 100
    assert list(p.coeffs) == [1, -3, 2]
    assert p.verify_real_rootedness()
    np.testing.assert_allclose(p.evaluate_roots_float64(), [1, 2])


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("constructor", ["coefficients", "roots", "recurrence"])
def test_root_arrays_protect_the_cache_and_allow_editable_copies(
    constructor: str, exact: bool
) -> None:
    if constructor == "recurrence":
        p = legendre_polynomial(2)
        expected = np.array([-1, 1]) / np.sqrt(3)
    elif constructor == "roots":
        p = RealRootedPolynomial.from_roots([-1, 1])
        expected = np.array([-1, 1])
    else:
        p = RealRootedPolynomial([1, 0, -1])
        expected = np.array([-1, 1])
    roots = p.evaluate_roots_float64(exact=exact)
    with pytest.raises(ValueError, match="read-only"):
        roots[0] = 100
    editable = roots.copy()
    editable[0] = 100
    np.testing.assert_allclose(p.evaluate_roots_float64(exact=exact), expected)
    assert roots is p.evaluate_roots_float64(exact=exact)


@pytest.mark.parametrize("ambient", [2, 4])
def test_normalized_arrays_protect_reconstruction_and_caches(ambient: int) -> None:
    p = RealRootedPolynomial.from_roots([1, 2])
    coefficients = p.normalized_coeffs(ambient)
    original = list(coefficients)
    with pytest.raises(ValueError, match="read-only"):
        coefficients[1] = 100
    editable = coefficients.copy()
    editable[1] = 100
    assert list(p.normalized_coeffs(ambient)) == original
    reconstructed = RealRootedPolynomial.from_normalized_coeffs(coefficients)
    expected_roots = [0] * (ambient - 2) + [1, 2]
    np.testing.assert_allclose(reconstructed.evaluate_roots_float64(), expected_roots)


def test_symbolic_normalized_array_is_readonly() -> None:
    p = RealRootedPolynomial([1, -sp.sqrt(2)])
    coefficients = p.normalized_coeffs()
    with pytest.raises(ValueError, match="read-only"):
        coefficients[1] = sp.I
    assert list(p.normalized_coeffs()) == [1, sp.sqrt(2)]


def test_unitary_roots_and_coefficients_do_not_expose_mutable_caches() -> None:
    source = np.array([1, 0, 1], dtype=object)
    p = UnitaryPolynomial(source)
    source[-1] = 100
    roots = p.evaluate_roots_float64()
    np.testing.assert_allclose(roots, [-1j, 1j], atol=1e-14)
    with pytest.raises(ValueError, match="read-only"):
        roots[0] = 100
    exposed = p.coeffs
    exposed[-1] = 100
    assert list(p.coeffs) == [1, 0, 1]
    assert roots is p.evaluate_roots_float64()


def test_t_transform_public_coefficients_cannot_mutate_the_polynomial_cache() -> None:
    p = RealRootedPolynomial.from_roots([1, 2])
    transform = FiniteTTransform(p)
    transform.e_k[1] = flint.fmpq(100)
    assert p._normalized_coeffs_flint() == [1, flint.fmpq(3, 2), 2]
    assert FiniteRTransform(p, order=2) == [sp.Rational(3, 2), sp.Rational(1, 2)]
