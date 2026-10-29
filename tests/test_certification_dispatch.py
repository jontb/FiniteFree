from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import PrecisionContext, RealRootedPolynomial, hahn_polynomial
from finitefree.core import UnitaryPolynomial
from finitefree.ensembles import wishart_expected_poly
from tests._cases import case


def test_medium_squarefree_factors_reuse_a_certified_arb_root_cache() -> None:
    degree: int
    for degree in [15, 16, 20, 30]:
        with case(degree=degree):
            expected = np.arange(-degree // 2, -degree // 2 + degree, dtype=float)
            coefficients = RealRootedPolynomial.from_roots(expected)._fmpq_poly
            p = RealRootedPolynomial(coefficients)
            with PrecisionContext(degree=degree, prec=128):
                initial_precision = flint.ctx.prec
                assert p.verify_real_rootedness()
                assert flint.ctx.prec == initial_precision
                assert p._roots_cached_exact
                assert p._roots_cached_prec == initial_precision
                roots = p.evaluate_roots_float64()
                np.testing.assert_array_equal(roots, expected)
                assert not roots.flags.writeable


def test_medium_degree_nearly_real_conjugate_pairs_are_rejected() -> None:
    degree: int
    for degree in [15, 16, 20, 30]:
        with case(degree=degree):
            real = RealRootedPolynomial.from_roots(range(1, degree - 1))._fmpq_poly
            # The imaginary parts are 10^-40, much smaller than a float tolerance.
            coefficients = real * flint.fmpq_poly([flint.fmpq(1, 10**80), 0, 1])
            p = RealRootedPolynomial(coefficients)
            with pytest.raises(ValueError, match="not real-rooted"):
                p.verify_real_rootedness()
            assert not p._is_verified
            assert p._roots_cached is None


@pytest.mark.parametrize("complex_pair", [False, True])
def test_unavailable_medium_degree_arb_certificate_falls_back_to_exact_sturm(
    complex_pair: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_roots = range(1, 15) if complex_pair else range(1, 17)
    coefficients = RealRootedPolynomial.from_roots(real_roots)._fmpq_poly
    if complex_pair:
        coefficients *= flint.fmpq_poly([1, 0, 1])
    p = RealRootedPolynomial(coefficients)
    calls = []

    def unavailable() -> bool:
        calls.append(True)
        raise RuntimeError("Backend could not isolate roots")

    monkeypatch.setattr(p, "_verify_real_rootedness_arb", unavailable)
    if complex_pair:
        with pytest.raises(ValueError, match="not real-rooted"):
            p.verify_real_rootedness()
        assert not p._is_verified
    else:
        assert p.verify_real_rootedness()
    assert calls == [True]
    assert p._roots_cached is None


def test_repeated_small_factors_keep_exact_sturm_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = RealRootedPolynomial(flint.fmpq_poly([-3, 1]) ** 30)

    def forbidden() -> Any:
        raise AssertionError("A small squarefree factor did not need Arb")

    monkeypatch.setattr(p, "_verify_real_rootedness_arb", forbidden)
    assert p.verify_real_rootedness()
    assert p._roots_cached is None
    np.testing.assert_array_equal(p.evaluate_roots_float64(), np.full(30, 3.0))


def test_assumed_real_roots_still_bypass_certification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = RealRootedPolynomial(flint.fmpq_poly([-1, 1]) ** 30, assume_real_rooted=True)

    def forbidden() -> Any:
        raise AssertionError("The explicit assumption should bypass certification")

    monkeypatch.setattr(p, "_verify_real_rootedness_arb", forbidden)
    assert p.verify_real_rootedness()


def _real_root_count(coefficients: Any) -> int:
    # SymPy's Sturm-based count is independent of the library's certification.
    return len(sp.real_roots(sp.Poly(list(coefficients), sp.Symbol("x"))))


def test_factories_trust_real_roots_only_in_guaranteed_domains() -> None:
    x = sp.Symbol("x")
    # KLS 9.5.1: Q_n(x; a, b, N) = 3F2(-n, n + a + b + 1, -x; a + 1, -N; 1).
    n, a, b, N = 4, sp.Rational(-5, 2), sp.Rational(-3, 2), 7
    hahn_sum = sum(
        sp.rf(-n, k)
        * sp.rf(n + a + b + 1, k)
        * sp.rf(-x, k)
        / (sp.rf(a + 1, k) * sp.rf(-N, k) * sp.factorial(k))
        for k in range(n + 1)
    )
    outside = hahn_polynomial(n, a, b, N)
    assert list(outside.coeffs) == sp.Poly(hahn_sum, x).monic().all_coeffs()
    assert _real_root_count(outside.coeffs) == 2
    assert not outside._is_verified
    with pytest.raises(ValueError, match="not real-rooted"):
        outside.verify_real_rootedness()

    inside = hahn_polynomial(4, sp.Rational(-1, 2), sp.Rational(3, 2), 7)
    assert inside._is_verified and _real_root_count(inside.coeffs) == 4
    with pytest.raises(ValueError, match="N must be an integer"):
        hahn_polynomial(2, 1, 1, 2.5)  # type: ignore[arg-type]

    # One complex sample column has E det(x - X X^*) = x^3 - E|X|^2 x^2.
    rank_one = wishart_expected_poly(3, 1)
    assert list(rank_one.coeffs) == [1, -3, 0, 0]
    np.testing.assert_array_equal(rank_one.evaluate_roots_float64(), [0, 0, 3])
    shape: tuple[int, int]
    for shape in [(3, 0), (3, -2), (-1, 2)]:
        with case(shape=shape):
            with pytest.raises(ValueError, match="d >= 0 and n >= 1"):
                wishart_expected_poly(*shape)


def test_unit_circle_trust_does_not_certify_derived_real_root_objects() -> None:
    u = UnitaryPolynomial([1, -1, 1, -1])  # roots 1, i, -i
    # p' = 3x^2 - 2x + 1 has roots (1 +/- i sqrt(2))/3; p(x - 1) has roots 2, 1 +/- i.
    derived: dict[str, RealRootedPolynomial] = {
        "derivative": u.derivative(),
        "shift": u.shift(1),
        "dilation": u.dilation(2),
        "projection": u.projection(2),
    }
    for name, p in derived.items():
        with case(transform=name):
            assert _real_root_count(p.coeffs) < p.degree
            assert not p._is_verified
            with pytest.raises(ValueError, match="not real-rooted"):
                p.evaluate_roots_float64()
    with pytest.raises(ValueError, match="requires a real-rooted"):
        u.verify_root_interlacing()
