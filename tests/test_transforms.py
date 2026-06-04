from typing import Iterator
from typing import List as TList

import numpy as np
import pytest
import sympy as sp

from finitefree.core import RealRootedPolynomial
from finitefree.transforms import (
    FiniteCauchyTransform,
    FiniteRTransform,
    FiniteSTransform,
    FiniteTTransform,
    SymmetricFiniteSTransform,
)


def test_finite_cauchy_transform() -> None:
    # p(z) = z^2 - 1
    p = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)
    G = FiniteCauchyTransform(p)
    z = sp.Symbol("z")
    # G(z) = 1/2 * (2z) / (z^2 - 1) = z / (z^2 - 1)
    val = sp.simplify(G - z / (z**2 - 1))
    assert val == 0


def test_finite_s_transform() -> None:
    # p(x) = x^2 - 3x + 2 = (x-1)(x-2)
    # roots are 1, 2 (strictly positive)
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    # c_0 = 1, c_1 = -3, c_2 = 2
    # e_0 = 1
    # -3 = -2 e_1 => e_1 = 3/2
    # 2 = 1 e_2 => e_2 = 2

    S = FiniteSTransform(p)
    assert len(S) == 2
    assert np.isclose(float(S[0]), 1 / (3 / 2))  # e_0 / e_1 = 2/3
    assert np.isclose(float(S[1]), (3 / 2) / 2)  # e_1 / e_2 = 3/4


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


def test_finite_r_transform() -> None:
    # Basic test for cumulant extraction
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    cumulants = FiniteRTransform(p, order=2)
    assert len(cumulants) == 2


def test_cumulant_additivity() -> None:
    from finitefree.convolutions import symmetric_additive

    # p: roots 1, 2, 3, 4
    p = RealRootedPolynomial([1, -10, 35, -50, 24], assume_real_rooted=True)
    # q: roots -3, -1, 1, 3
    q = RealRootedPolynomial([1, 0, -10, 0, 9], assume_real_rooted=True)

    d = 4
    r = symmetric_additive(p, q, d)

    cum_p = FiniteRTransform(p, order=d)
    cum_q = FiniteRTransform(q, order=d)
    cum_r = FiniteRTransform(r, order=d)

    for i in range(d):
        assert np.isclose(float(cum_r[i]), float(cum_p[i]) + float(cum_q[i])), (
            f"Cumulant {i + 1} mismatch: {cum_r[i]} != {cum_p[i]} + {cum_q[i]}"
        )


def test_finite_free_cumulants_definition_consistency() -> None:
    # A partition generator to compute the definition directly
    def get_partitions(elements: TList[int]) -> Iterator[TList[TList[int]]]:
        if not elements:
            yield []
            return
        first = elements[0]
        for sub in get_partitions(elements[1:]):
            for i, block in enumerate(sub):
                yield sub[:i] + [[first] + block] + sub[i + 1 :]
            yield sub + [[first]]

    # p(x) = (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
    p = RealRootedPolynomial([1, -6, 11, -6], assume_real_rooted=True)
    d = 3
    e_k = p.normalized_coeffs(d)

    # Compute using Definition 2.14:
    # κ_n^{(d)}(p) = (-d)^{n-1} (n-1)! \sum_{\pi} ...
    import math

    for n in range(1, 4):
        elements = list(range(1, n + 1))
        sum_val = 0.0
        for partition in get_partitions(elements):
            num_blocks = len(partition)
            coeff = ((-1) ** (num_blocks - 1)) * math.factorial(num_blocks - 1)
            prod = 1.0
            for block in partition:
                prod *= float(e_k[len(block)])
            sum_val += coeff * prod
        expected = sum_val * ((-d) ** (n - 1)) * math.factorial(n - 1)

        # Compare with FiniteRTransform
        cumulants = FiniteRTransform(p, order=n, d=d)
        assert np.isclose(float(cumulants[-1]), expected)


def test_projection_cumulant_relation() -> None:
    # p(x) = (x-1)(x-2)(x-3)(x-4)
    p = RealRootedPolynomial([1, -10, 35, -50, 24], assume_real_rooted=True)
    d = 4
    j = 3

    # LHS: κ_n^{(j)}(\partial^{j|d} p)
    proj_p = p.projection(j)

    # RHS: κ_n^{(d)}([Dil_{j/d} p]^{\boxplus_d d/j})
    # Dilate by j/d = 3/4
    dil_p = p.dilation(sp.Rational(j, d))
    # Convolution power by d/j = 4/3
    rhs_poly = dil_p.additive_power(sp.Rational(d, j))

    # Compare cumulants for 1 <= n <= j
    for n in range(1, j + 1):
        lhs_cum = FiniteRTransform(proj_p, order=n, d=j)[-1]
        rhs_cum = FiniteRTransform(rhs_poly, order=n, d=d)[-1]
        assert np.isclose(float(lhs_cum), float(rhs_cum))


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

    # Non-symmetric polynomial test
    p_non_sym = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    assert p_non_sym.is_symmetric() is False
    with pytest.raises(ValueError):
        p_non_sym.square_roots_map()
    with pytest.raises(ValueError):
        SymmetricFiniteSTransform(p_non_sym)
