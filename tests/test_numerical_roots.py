from typing import Any, Callable

import numpy as np
import pytest
import sympy as sp

from finitefree import (
    PrecisionContext,
    RealRootedPolynomial,
    chebyshev_t_polynomial,
    chebyshev_u_polynomial,
    gue_expected_poly,
    hermite_polynomial,
    jacobi_polynomial,
    laguerre_polynomial,
    legendre_polynomial,
    wishart_expected_poly,
)
from finitefree.convolutions import multiplicative, symmetric_additive


def reference_roots(p: RealRootedPolynomial) -> Any:
    with PrecisionContext(degree=p.degree, prec=192):
        pairs: Any = p._fmpq_poly.complex_roots()
        assert all(root.imag.is_zero() for root, _ in pairs)
        return np.sort(
            [
                float(root.real)
                for root, multiplicity in pairs
                for _ in range(multiplicity)
            ]
        )


@pytest.mark.parametrize(
    "factory",
    [
        hermite_polynomial,
        lambda n: hermite_polynomial(n, False),
        legendre_polynomial,
        chebyshev_t_polynomial,
        chebyshev_u_polynomial,
        lambda n: laguerre_polynomial(n, 0),
        lambda n: jacobi_polynomial(n, sp.Rational(-1, 2), sp.Rational(-1, 2)),
        lambda n: jacobi_polynomial(n, sp.Rational(-3, 4), sp.Rational(3, 2)),
        gue_expected_poly,
        lambda n: wishart_expected_poly(n, 2 * n),
    ],
)
@pytest.mark.parametrize("degree", [2, 17, 64])
def test_recurrence_roots_against_independent_arb(
    factory: Callable[[int], RealRootedPolynomial], degree: int
) -> None:
    p = factory(degree)
    assert p._root_recurrence is not None
    expected = reference_roots(p)
    actual = p.evaluate_roots_float64(exact=False)
    np.testing.assert_allclose(
        actual, expected, atol=3e-14 * max(1, max(abs(expected))), rtol=3e-13
    )


def test_laguerre_tiny_positive_hard_edge() -> None:
    p = laguerre_polynomial(64, -1 + sp.Rational(1, 10**12))
    expected = reference_roots(p)
    actual = p.evaluate_roots_float64(exact=False)
    assert 0 < actual[0] < 1e-13
    assert abs(actual[0] / expected[0] - 1) < 2e-10


@pytest.mark.parametrize("scale", [sp.Rational(1, 10**100), 10**100, -3])
def test_affine_metadata_stays_consistent(scale: Any) -> None:
    p = hermite_polynomial(17, False).dilation(scale).shift(scale / 7)
    expected = reference_roots(p)
    actual = p.evaluate_roots_float64(exact=False)
    np.testing.assert_allclose(
        actual / float(scale), expected / float(scale), atol=3e-13
    )


def test_reference_request_does_not_reuse_numerical_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = legendre_polynomial(17)
    calls = []
    original = p._evaluate_roots_float64_uncached

    def observe(parallel: bool = False, exact: bool = True) -> Any:
        calls.append(exact)
        return original(parallel=parallel, exact=exact)

    monkeypatch.setattr(p, "_evaluate_roots_float64_uncached", observe)
    p.evaluate_roots_float64(exact=False)
    p.evaluate_roots_float64(exact=True)
    p.evaluate_roots_float64(exact=False)
    assert calls == [False, True]


def test_affine_parameters_cancel_before_float_conversion() -> None:
    p = (
        legendre_polynomial(17)
        .shift(10**20)
        .shift(-(10**20) + 1)
        .dilation(10**20 + 1)
        .dilation(sp.Rational(1, 10**20 + 1))
    )
    np.testing.assert_allclose(
        p.evaluate_roots_float64(exact=False), reference_roots(p), atol=2e-15
    )


def test_only_proven_hermite_convolutions_keep_recurrence() -> None:
    p = gue_expected_poly(40).shift(2).dilation(sp.Rational(3, 2))
    q = hermite_polynomial(40, True).shift(-1)
    result = symmetric_additive(p, q, 40)
    assert result._root_recurrence is not None
    np.testing.assert_allclose(
        result.evaluate_roots_float64(exact=False), reference_roots(result), atol=5e-13
    )
    generic = multiplicative(p, RealRootedPolynomial.from_roots(range(1, 41)), 40)
    assert generic._root_recurrence is None


def test_outside_orthogonality_regime_has_no_recurrence() -> None:
    assert laguerre_polynomial(5, -2)._root_recurrence is None
    assert jacobi_polynomial(5, -2, 3)._root_recurrence is None


@pytest.mark.parametrize("p", [laguerre_polynomial(2, -3), jacobi_polynomial(2, -3, 2)])
def test_nonorthogonal_complex_roots_are_rejected(p: RealRootedPolynomial) -> None:
    with pytest.raises(ValueError, match="not real-rooted"):
        p.evaluate_roots_float64(exact=False)


def test_unrepresentable_recurrence_retains_reference_path() -> None:
    p = laguerre_polynomial(2, -1 + sp.Rational(1, 10**400))
    assert p._root_recurrence is None
    # The exact coefficients still distinguish alpha from -1.
    assert p._fmpq_poly[0] != 0


def test_tridiagonal_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import scipy.linalg

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise np.linalg.LinAlgError("LAPACK did not converge")

    monkeypatch.setattr(scipy.linalg, "eigvalsh_tridiagonal", fail)
    p = legendre_polynomial(5)
    np.testing.assert_allclose(
        p.evaluate_roots_float64(exact=False), reference_roots(p), atol=1e-13
    )
