import numpy as np
import pytest

from finitefree.core import RealRootedPolynomial


def test_dilation() -> None:
    # p(x) = (x - 1)(x - 2) = x^2 - 3x + 2
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    assert p._is_verified

    # Dilate by 2: [Dil_2 p](x) = 2^2 p(x/2) = x^2 - 6x + 8
    p_dil = p.dilation(2)
    assert np.allclose(list(p_dil.coeffs), [1, -6, 8])
    assert p_dil._is_verified

    # Dilate by -1: [Dil_-1 p](x) = (-1)^2 p(-x) = x^2 + 3x + 2
    p_dil_neg = p.dilation(-1)
    assert np.allclose(list(p_dil_neg.coeffs), [1, 3, 2])
    assert p_dil_neg._is_verified

    # Error on zero dilation factor
    with pytest.raises(ValueError, match="Dilation factor c cannot be zero"):
        p.dilation(0)


def test_shift() -> None:
    # p(x) = (x - 1)(x - 2) = x^2 - 3x + 2
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    assert p._is_verified

    # Shift by 1: [Shi_1 p](x) = p(x-1) = (x-2)(x-3) = x^2 - 5x + 6
    p_shift = p.shift(1)
    assert np.allclose(list(p_shift.coeffs), [1, -5, 6])
    assert p_shift._is_verified

    # Shift by -2: [Shi_-2 p](x) = p(x+2) = (x+1)x = x^2 + x
    p_shift_neg = p.shift(-2)
    assert np.allclose(list(p_shift_neg.coeffs), [1, 1, 0])
    assert p_shift_neg._is_verified


def test_power() -> None:
    # p(x) = (x - 1)(x - 2) = x^2 - 3x + 2 (roots 1, 2)
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)

    # Power by 2: roots become 1^2 = 1, 2^2 = 4 => (x - 1)(x - 4) = x^2 - 5x + 4
    p_pow = p.power(2)
    assert np.allclose(list(p_pow.coeffs), [1, -5, 4])

    # Power by 0.5: roots become 1^0.5 = 1, 2^0.5 = sqrt(2)
    p_pow_half = p.power(0.5)
    roots = p_pow_half.evaluate_roots_float64()
    assert np.allclose(roots, [1.0, np.sqrt(2)])

    # Error on non-positive power
    with pytest.raises(ValueError, match="Power factor c must be strictly positive"):
        p.power(0)

    with pytest.raises(ValueError, match="Power factor c must be strictly positive"):
        p.power(-1)

    # Error on negative roots
    p_neg = RealRootedPolynomial([1, 2, 1], assume_real_rooted=True)
    with pytest.raises(
        ValueError, match="only defined for polynomials with non-negative roots"
    ):
        p_neg.power(2)


def test_reversed_polynomial() -> None:
    # p(x) = (x - 1)(x - 2) = x^2 - 3x + 2 (roots 1, 2)
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)

    # Reversed roots: 1, 0.5 => (x - 1)(x - 0.5) = x^2 - 1.5x + 0.5
    p_rev = p.reversed_polynomial()
    assert np.allclose(list(p_rev.coeffs), [1, -1.5, 0.5])

    # Try reversing a polynomial with a root at zero
    # p(x) = x(x - 1) = x^2 - x
    p_zero = RealRootedPolynomial([1, -1, 0], assume_real_rooted=True)
    with pytest.raises(ValueError, match="strictly non-zero roots"):
        p_zero.reversed_polynomial()


def test_phi_d() -> None:
    # 1. p(x) = (x - 1)(x - 2) = x^2 - 3x + 2
    # Roots: 1, 2. r = 0.
    # e_tilde: e_0=1, e_1=1.5, e_2=2
    # Phi_2(p) roots: 1.5, 4/3 = 1.33333...
    p1 = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    phi1 = p1.phi_d()
    roots1 = phi1.evaluate_roots_float64()
    assert np.allclose(roots1, [4 / 3, 1.5])

    # 2. p(x) = x(x - 2) = x^2 - 2x
    # Roots: 0, 2. r = 1.
    # e_tilde: e_0=1, e_1=1, e_2=0
    # Phi_2(p) roots: e_1/e_0 = 1, and 0
    p2 = RealRootedPolynomial([1, -2, 0], assume_real_rooted=True)
    phi2 = p2.phi_d()
    roots2 = phi2.evaluate_roots_float64()
    assert np.allclose(roots2, [0.0, 1.0])

    # 3. Negative roots test (should raise ValueError)
    p3 = RealRootedPolynomial([1, 2, 1], assume_real_rooted=True)  # (x+1)^2
    with pytest.raises(
        ValueError, match="only defined for polynomials with non-negative roots"
    ):
        p3.phi_d()
