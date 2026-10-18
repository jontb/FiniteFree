import numpy as np
import pytest
import sympy as sp

from finitefree.core import RealRootedPolynomial
from finitefree.transforms import (
    FiniteCauchyTransform,
    FiniteSTransform,
    FiniteTTransform,
    SymmetricFiniteSTransform,
)
from tests._cases import case
from tests._cumulants import assert_cumulants


def test_cauchy_transform_against_exact_root_sum() -> None:
    z = sp.Symbol("z")
    for roots in ([-1, 1], [0, 1, 3]):
        with case(roots=roots):
            p = RealRootedPolynomial.from_roots(roots)
            expected = sum(sp.Rational(1, len(roots)) / (z - root) for root in roots)
            actual = FiniteCauchyTransform(p)
            assert not actual.atoms(sp.Float)
            assert sp.cancel(actual - expected) == 0
    with pytest.raises(ValueError, match="positive degree"):
        FiniteCauchyTransform(RealRootedPolynomial([1]))


def test_finite_s_transform() -> None:
    # p(x) = x^2 - 3x + 2 = (x-1)(x-2)
    # roots are 1, 2 (strictly positive)
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    # c_0 = 1, c_1 = -3, c_2 = 2
    # e_0 = 1
    # -3 = -2 e_1 => e_1 = 3/2
    # 2 = 1 e_2 => e_2 = 2

    S = FiniteSTransform(p)
    assert list(S) == [sp.Rational(2, 3), sp.Rational(3, 4)]
    assert len(S) == 2
    assert np.isclose(float(S[0]), 1 / (3 / 2))  # e_0 / e_1 = 2/3
    assert np.isclose(float(S[1]), (3 / 2) / 2)  # e_1 / e_2 = 3/4

    # Test exact=False float output
    S_float = FiniteSTransform(p, exact=False)
    assert S_float.dtype == np.float64
    assert np.allclose(S_float, [2 / 3, 3 / 4])


def test_lazy_geometric_properties() -> None:
    # 1. Polynomial containing a root at zero
    # p(x) = x^2 (roots 0, 0) -> [1, 0, 0]
    p_zero = RealRootedPolynomial([1, 0, 0], assume_real_rooted=True)
    assert p_zero.has_non_negative_roots is True
    assert p_zero.has_strictly_positive_roots is False

    # 2. Polynomial with strictly negative roots
    # p(x) = (x+1)(x+2) = x^2 + 3x + 2 -> [1, 3, 2]
    p_neg = RealRootedPolynomial([1, 3, 2], assume_real_rooted=True)
    assert p_neg.has_non_negative_roots is False
    assert p_neg.has_strictly_positive_roots is False

    # 3. Valid positive-rooted polynomial
    # p(x) = (x-1)(x-2) = x^2 - 3x + 2 -> [1, -3, 2]
    p_pos = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    assert p_pos.has_non_negative_roots is True
    assert p_pos.has_strictly_positive_roots is True


def test_finite_s_transform_errors() -> None:
    from finitefree import UnitaryPolynomial

    # 1. Real rooted polynomial with negative roots -> fails has_strictly_positive_roots
    p_neg = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)
    with pytest.raises(ValueError, match="roots must be strictly positive"):
        FiniteSTransform(p_neg)

    # 2. Unitary polynomial with zero coefficient -> fails the zero coefficient guard
    p_unit = UnitaryPolynomial([1, 0, -1])
    with pytest.raises(ValueError, match="a zero coefficient was encountered"):
        FiniteSTransform(p_unit)


def test_additive_convolution_and_powers_preserve_exact_cumulants() -> None:
    from finitefree.convolutions import symmetric_additive

    pairs = [
        (
            RealRootedPolynomial([1, -10, 35, -50, 24], assume_real_rooted=True),
            RealRootedPolynomial([1, 0, -10, 0, 9], assume_real_rooted=True),
        ),
        (
            RealRootedPolynomial.from_roots([0, 0, 3]),
            RealRootedPolynomial.from_roots([0, 0, 3]),
        ),
        (
            RealRootedPolynomial([2, -6, 4], monic=False),
            RealRootedPolynomial([-3, 9, -6], monic=False),
        ),
    ]
    for p, q in pairs:
        with case(coefficients=list(p.coeffs)):
            d = p.degree
            a, b = assert_cumulants(p, d), assert_cumulants(q, d)
            result = symmetric_additive(p, q, d)
            assert_cumulants(result, d, expected=[x + y for x, y in zip(a, b)])
            monic = [sp.cancel(c / p.coeffs[0]) for c in p.coeffs]
            for power in (1, 2, sp.Rational(3, 2)):
                powered = p.additive_power(power)
                assert_cumulants(powered, d, expected=[power * x for x in a])
                if power == 1:
                    assert list(powered.coeffs) == monic
                if power == 2:
                    assert list(powered.coeffs) == list(
                        symmetric_additive(p, p, d).coeffs
                    )
            if d == 4:
                assert a == [
                    sp.Rational(5, 2),
                    sp.Rational(5, 3),
                    0,
                    sp.Rational(-4, 9),
                ]
            elif d == 3:
                assert a == [1, 3, 9]
                assert list(p.additive_power(sp.Rational(3, 2)).coeffs) == [
                    1,
                    sp.Rational(-9, 2),
                    sp.Rational(9, 4),
                    sp.Rational(3, 8),
                ]
                assert (
                    list(p.additive_power(sp.Rational(1, 2)).additive_power(2).coeffs)
                    == monic
                )
            else:
                assert a == [sp.Rational(3, 2), sp.Rational(1, 2)]
                assert list(p.additive_power(2).coeffs) == [1, -6, sp.Rational(17, 2)]


def test_finite_t_transform() -> None:
    # 1. Test polynomial with strictly positive roots (r=0)
    # p(x) = (x-1)(x-2)(x-3)(x-4)
    p = RealRootedPolynomial([1, -10, 35, -50, 24], assume_real_rooted=True)
    d = 4
    T = FiniteTTransform(p)
    assert T.r == 0

    # Verify T_p(d)( (d-k)/d ) = 1 / S_p(d)( -k/d ) for k = 1, ..., d
    S = FiniteSTransform(p)
    for k in range(1, d + 1):
        t_val = (d - k) / d
        if t_val > 0:
            val_T = T(t_val)
            val_S_inv = 1 / S[k - 1]
            assert np.isclose(float(val_T), float(val_S_inv))

    # Test boundary/invalid t
    with pytest.raises(ValueError, match="must be in the open interval"):
        T(0)
    with pytest.raises(ValueError, match="must be in the open interval"):
        T(1)
    with pytest.raises(ValueError, match="must be in the open interval"):
        T(1.5)

    # 2. Test polynomial with zero root (r > 0)
    # q(x) = x^2 (x-1)(x-2) = x^4 - 3x^3 + 2x^2
    # roots: 0, 0, 1, 2. degree = 4, r = 2.
    q = RealRootedPolynomial([1, -3, 2, 0, 0], assume_real_rooted=True)
    T_q = FiniteTTransform(q)
    assert T_q.r == 2

    # For t in (0, r/d) -> (0, 0.5), T(t) should be 0
    assert T_q(0.1) == 0
    assert T_q(0.49) == 0

    # For t in [2/4, 3/4) = [0.5, 0.75) -> k=3
    # value is e_tilde_{4-3+1} / e_tilde_{4-3} = e_tilde_2 / e_tilde_1
    e_k = q.normalized_coeffs()
    val_k3 = e_k[2] / e_k[1]
    assert np.isclose(float(T_q(0.5)), float(val_k3))
    assert np.isclose(float(T_q(0.7)), float(val_k3))

    # For t in [3/4, 1) = [0.75, 1) -> k=4
    # value is e_tilde_{4-4+1} / e_tilde_{4-4} = e_tilde_1 / e_tilde_0
    val_k4 = e_k[1] / e_k[0]
    assert np.isclose(float(T_q(0.75)), float(val_k4))
    assert np.isclose(float(T_q(0.99)), float(val_k4))

    # 3. Test validation for negative roots
    with pytest.raises(
        ValueError, match="only defined for polynomials with non-negative roots"
    ):
        # roots: -1, 1
        p_neg = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)
        FiniteTTransform(p_neg)


def test_symmetric_polynomial_and_s_transform() -> None:
    # p(x) = (x^2 - 1)(x^2 - 4) = x^4 - 5x^2 + 4
    # roots: -2, -1, 1, 2. degree = 4, symmetric.
    p = RealRootedPolynomial([1, 0, -5, 0, 4], assume_real_rooted=True)
    assert p.is_symmetric() is True

    # Test Sq map
    # Sq(p)(y) = (y - 1)(y - 4) = y^2 - 5y + 4
    sq_p = p.square_roots_map()
    assert sq_p.degree == 2
    assert sq_p.coeffs[0] == 1
    assert sq_p.coeffs[1] == -5
    assert sq_p.coeffs[2] == 4

    # Test SymmetricFiniteSTransform
    # e_tilde for p:
    # e_0 = 1
    # e_1 = 0
    # e_2 = -5 / (-1)^2 * C(4, 2) = -5 / 6
    # e_3 = 0
    # e_4 = 4 / (-1)^4 * C(4, 4) = 4
    # Symmetric S-transform on k=1, 2:
    # For k=1: e_tilde_0 / e_tilde_2 = 1 / (-5/6) = -6/5
    # For k=2: e_tilde_2 / e_tilde_4 = (-5/6) / 4 = -5/24
    s_sym = SymmetricFiniteSTransform(p)
    assert len(s_sym) == 2
    assert s_sym[0] == sp.Rational(-6, 5)
    assert s_sym[1] == sp.Rational(-5, 24)

    # Test exact=False float output
    s_sym_float = SymmetricFiniteSTransform(p, exact=False)
    assert s_sym_float.dtype == np.float64
    assert np.allclose(s_sym_float, [-6 / 5, -5 / 24])

    # Non-symmetric polynomial test
    p_non_sym = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    assert p_non_sym.is_symmetric() is False
    with pytest.raises(ValueError):
        p_non_sym.square_roots_map()
    with pytest.raises(ValueError):
        SymmetricFiniteSTransform(p_non_sym)


def test_coefficient_signs_do_not_certify_positive_roots() -> None:
    # x^2 - x + 1 has alternating coefficients but no real roots.
    with pytest.raises(ValueError, match="not real-rooted"):
        FiniteSTransform(RealRootedPolynomial([1, -1, 1]))
    with pytest.raises(ValueError, match="not real-rooted"):
        FiniteTTransform(RealRootedPolynomial([1, -1, 1]))
