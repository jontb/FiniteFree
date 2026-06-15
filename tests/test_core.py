import flint
import numpy as np
import pytest

from finitefree import PrecisionContext, RealRootedPolynomial


def test_precision_context() -> None:
    initial_prec = flint.ctx.prec
    with PrecisionContext(degree=100):
        assert flint.ctx.prec >= 53
        assert flint.ctx.prec > initial_prec
    assert flint.ctx.prec == initial_prec

    # Test custom precision override
    with PrecisionContext(degree=10, prec=128):
        assert flint.ctx.prec == 128
    assert flint.ctx.prec == initial_prec


def test_real_rooted_polynomial_monic() -> None:
    # 2x^2 - 8 -> x^2 - 4
    poly = RealRootedPolynomial([2, 0, -8])
    assert np.allclose(list(poly.coeffs), [1, 0, -4])
    assert poly.degree == 2


def test_verify_real_rootedness_success() -> None:
    # x^2 - 3x + 2 = (x-1)(x-2)
    poly = RealRootedPolynomial([1, -3, 2])
    assert poly.verify_real_rootedness()


def test_verify_real_rootedness_failure() -> None:
    # x^2 + 1
    poly = RealRootedPolynomial([1, 0, 1])
    with pytest.raises(ValueError, match="not real-rooted"):
        poly.verify_real_rootedness()


def test_normalized_coeffs() -> None:
    # p(x) = x^2 - 2x + 1 = (x-1)^2
    # c_0 = 1, c_1 = -2, c_2 = 1
    # c_k = (-1)^k \binom{2}{k} e_k
    # k=0: c_0 = 1 * 1 * e_0 => e_0 = 1
    # k=1: c_1 = -1 * 2 * e_1 => -2 = -2 e_1 => e_1 = 1
    # k=2: c_2 = 1 * 1 * e_2 => 1 = 1 e_2 => e_2 = 1
    poly = RealRootedPolynomial([1, -2, 1], assume_real_rooted=True)
    e_k = poly.normalized_coeffs()
    assert np.allclose(list(e_k), [1, 1, 1])


def test_from_normalized_coeffs() -> None:
    e_k = [1, 1, 1]
    poly = RealRootedPolynomial.from_normalized_coeffs(e_k)
    assert np.allclose(list(poly.coeffs), [1, -2, 1])
    assert poly._is_verified


def test_verify_root_interlacing() -> None:
    # p(x) = x^2 - 3x + 2, roots: 1, 2
    # p'(x) = 2x - 3, root: 1.5
    # 1 < 1.5 < 2
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    assert p.verify_root_interlacing(strict=True)

    # Double root: p(x) = x^2 - 2x + 1 = (x-1)^2
    # p'(x) = 2x - 2, root: 1
    p2 = RealRootedPolynomial([1, -2, 1], assume_real_rooted=True)
    # Should pass non-strict
    assert p2.verify_root_interlacing(strict=False)

    # Should fail strict
    with pytest.raises(ValueError, match="Strict root interlacing failed"):
        p2.verify_root_interlacing(strict=True)


def test_sturm_bypass_high_degree() -> None:
    # Construct a high-degree polynomial (d=35)
    roots = list(range(1, 36))
    p = RealRootedPolynomial.from_roots(roots)
    assert p.degree == 35
    assert p.verify_real_rootedness() is True


def test_coefficient_scaling_preconditioning() -> None:
    # Construct a polynomial with widely spread roots to test scaling
    roots = [1e-3, 1.0, 1e3]
    p = RealRootedPolynomial.from_roots(roots)
    eval_roots = p.evaluate_roots_float64(exact=False)
    np.testing.assert_allclose(eval_roots, sorted(roots), rtol=1e-5)


def _build_hermite_poly(d: int) -> RealRootedPolynomial:
    import math

    coeffs = [0] * (d + 1)
    for m in range(d // 2 + 1):
        num = math.factorial(d)
        den = math.factorial(m) * math.factorial(d - 2 * m) * (2**m)
        val = ((-1) ** m) * (num // den)
        coeffs[2 * m] = val
    return RealRootedPolynomial(coeffs, assume_real_rooted=True)


def test_hybrid_root_isolation() -> None:
    d = 50
    p = _build_hermite_poly(d)
    roots_seq = p.evaluate_roots_float64(parallel=False, exact=False)
    roots_par = p.evaluate_roots_float64(parallel=True, exact=False)
    assert len(roots_seq) == d
    assert len(roots_par) == d
    assert np.allclose(roots_seq, roots_par, atol=1e-8)


def test_from_roots() -> None:
    import sympy as sp

    # Test reconstruction of exact polynomial from roots
    roots = [sp.Rational(1, 2), sp.Rational(-2, 3), sp.Rational(3, 4)]
    p = RealRootedPolynomial.from_roots(roots)

    expected_coeffs = [1, sp.Rational(-7, 12), sp.Rational(-11, 24), sp.Rational(1, 4)]
    assert len(p.coeffs) == 4
    for c, exp_c in zip(p.coeffs, expected_coeffs):
        assert c == exp_c

    # Scaled check: Reconstruct a degree 50 polynomial from 50 distinct roots
    roots_scaled = [sp.Rational(i, 3) for i in range(-25, 25)]
    p_scaled = RealRootedPolynomial.from_roots(roots_scaled)
    assert p_scaled.degree == 50

    expected_poly = flint.fmpq_poly([1])
    for r in roots_scaled:
        expected_poly *= flint.fmpq_poly([-flint.fmpq(r.p, r.q), 1])

    flint_expected_coeffs = list(reversed(list(expected_poly)))
    assert len(p_scaled.coeffs) == len(flint_expected_coeffs)
    for c, exp_c in zip(p_scaled.coeffs, flint_expected_coeffs):
        assert c == sp.Rational(int(exp_c.p), int(exp_c.q))


def test_polynomial_representations() -> None:
    p = RealRootedPolynomial([1, -2, 1])
    assert str(p) == "RealRootedPolynomial(x**2 - 2*x + 1)"
    assert repr(p) == "RealRootedPolynomial(x**2 - 2*x + 1)"


def test_polynomial_boundary_cases() -> None:
    # Degree 0 polynomial: constant p(x) = 5 (normalized to 1)
    p0 = RealRootedPolynomial([5])
    assert p0.degree == 0
    assert np.allclose(list(p0.coeffs), [1])
    assert p0.verify_real_rootedness() is True
    assert len(p0.evaluate_roots_float64()) == 0

    # Degree 1 polynomial: p(x) = 2x - 3 (normalized to x - 1.5)
    p1 = RealRootedPolynomial([2, -3])
    assert p1.degree == 1
    assert np.allclose(list(p1.coeffs), [1, -1.5])
    assert p1.verify_real_rootedness() is True
    roots = p1.evaluate_roots_float64()
    assert len(roots) == 1
    assert np.isclose(roots[0], 1.5)

    # Empty coefficients should raise an error
    with pytest.raises((ValueError, IndexError)):
        RealRootedPolynomial([])


def test_exact_roots_default() -> None:
    # Roots reconstructed and evaluated should match exactly under default exact=True
    roots = [1.0, 2.0, 3.0]
    p = RealRootedPolynomial.from_roots(roots)

    # We clear the cached roots to make sure the evaluation path is executed
    p._roots_cached = None
    eval_roots_default = p.evaluate_roots_float64()

    p._roots_cached = None
    eval_roots_exact = p.evaluate_roots_float64(exact=True)

    p._roots_cached = None
    eval_roots_approx = p.evaluate_roots_float64(exact=False)

    np.testing.assert_allclose(eval_roots_default, roots)
    np.testing.assert_allclose(eval_roots_exact, roots)
    np.testing.assert_allclose(eval_roots_approx, roots)


def test_derivative_non_monic() -> None:
    # p(x) = x^2 - 3x + 2, derivative is 2x - 3
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    
    # Monic (default): returns x - 1.5
    dp_monic = p.derivative(monic=True)
    assert np.allclose(list(dp_monic.coeffs), [1, -1.5])
    
    # Non-monic: returns 2x - 3
    dp_non_monic = p.derivative(monic=False)
    assert np.allclose(list(dp_non_monic.coeffs), [2, -3])
