from typing import Any, Callable

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    FiniteRTransform,
    FiniteTTransform,
    RealRootedPolynomial,
    UnitaryPolynomial,
)
from tests._cases import case


def test_mutating_a_flint_input_cannot_change_coefficients_or_certificates() -> None:
    monic: bool
    warm_cache: bool
    for warm_cache in [True, False]:
        for monic in [True, False]:
            with case(monic=monic, warm_cache=warm_cache):
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


def test_coefficient_ownership_across_constructors_and_backends() -> None:
    scenarios: list[tuple[str, Any, Callable[[Any], RealRootedPolynomial]]] = [
        ("list", [1, -3, 2], RealRootedPolynomial),
        ("array", np.array([1, -3, 2], dtype=object), RealRootedPolynomial),
        ("symbolic", np.array([1, -sp.sqrt(2)], dtype=object), RealRootedPolynomial),
        ("roots", np.array([1, 2], dtype=object), RealRootedPolynomial.from_roots),
        (
            "normalized",
            np.array([1, sp.Rational(3, 2), 2], dtype=object),
            RealRootedPolynomial.from_normalized_coeffs,
        ),
    ]
    for name, source, constructor in scenarios:
        with case(constructor=name):
            p = constructor(source)
            expected = [1, -sp.sqrt(2)] if name == "symbolic" else [1, -3, 2]
            source[-1] = sp.I if name == "symbolic" else 100
            exported = p.coeffs
            exported[-1] = 100
            assert list(p.coeffs) == expected
            root = sp.sqrt(2) if name == "symbolic" else 1
            assert p.evaluate(root) == 0
            assert p.verify_real_rootedness()
            np.testing.assert_allclose(
                p.evaluate_roots_float64(),
                [np.sqrt(2)] if name == "symbolic" else [1, 2],
            )
            if name == "symbolic":
                with pytest.raises(ValueError, match="read-only"):
                    p.coeffs_sympy[-1] = sp.I
            else:
                assert FiniteRTransform(p, order=2) == [
                    sp.Rational(3, 2),
                    sp.Rational(1, 2),
                ]


def test_normalized_coefficient_ownership_through_reconstruction_and_transforms() -> (
    None
):
    for ambient in (2, 4, 1):
        with case(ambient=ambient):
            symbolic = ambient == 1
            p = (
                RealRootedPolynomial([1, -sp.sqrt(2)])
                if symbolic
                else RealRootedPolynomial.from_roots([1, 2])
            )
            coefficients = p.normalized_coeffs(ambient)
            original = list(coefficients)
            if symbolic:
                assert original == [1, sp.sqrt(2)]
            with pytest.raises(ValueError, match="read-only"):
                coefficients[1] = sp.I if symbolic else 100
            editable = coefficients.copy()
            editable[1] = 100
            assert list(p.normalized_coeffs(ambient)) == original
            reconstructed = RealRootedPolynomial.from_normalized_coeffs(coefficients)
            expected = [np.sqrt(2)] if symbolic else [0] * (ambient - 2) + [1, 2]
            np.testing.assert_allclose(reconstructed.evaluate_roots_float64(), expected)
            if not symbolic:
                transform = FiniteTTransform(p)
                transform.e_k[1] = flint.fmpq(100)
                assert p._normalized_coeffs_flint() == [1, flint.fmpq(3, 2), 2]
                assert FiniteRTransform(p, order=2) == [
                    sp.Rational(3, 2),
                    sp.Rational(1, 2),
                ]


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
