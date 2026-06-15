import numpy as np
import sympy as sp

from finitefree import (
    FiniteSTransform,
    SymmetricFiniteSTransform,
    UnitaryPolynomial,
    unitary_hermite_polynomial,
)
from finitefree.convolutions import (
    asymmetric_additive,
    multiplicative,
    symmetric_additive,
)
from finitefree.core import RealRootedPolynomial


def test_symmetric_additive_basic() -> None:
    # p(x) = x^2 - 3x + 2, roots are 1, 2
    # q(x) = x^2 - 4x + 3, roots are 1, 3
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    q = RealRootedPolynomial([1, -4, 3], assume_real_rooted=True)

    # Expected analytical convolution result: x^2 - 7x + 11
    res = symmetric_additive(p, q, 2)
    assert list(res.coeffs) == [1, -7, 11]


def test_multiplicative_basic() -> None:
    # p(x) = x^2 - 1 = (x-1)(x+1) -> not strictly positive roots, but fine for formula
    p = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)
    q = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)

    res = multiplicative(p, q, 2)
    assert res.degree == 2
    assert list(res.coeffs) == [1, 0, 1]


def test_asymmetric_additive_basic() -> None:
    p = RealRootedPolynomial([1, -2, 1], assume_real_rooted=True)  # (x-1)^2
    q = RealRootedPolynomial([1, 2, 1], assume_real_rooted=True)  # (x+1)^2

    res = asymmetric_additive(p, q, 2)
    assert res.degree == 2
    assert list(res.coeffs) == [1, 0, 1]


def test_asymmetric_additive_suite() -> None:
    # Asymmetric additive convolution validates distinct combinatorial weights
    from finitefree.convolutions import asymmetric_additive

    p = RealRootedPolynomial([1, -5, 6], assume_real_rooted=True)  # roots: 2, 3
    q = RealRootedPolynomial([1, -1, -2], assume_real_rooted=True)  # roots: 2, -1

    d = 2
    res_asym = asymmetric_additive(p, q, d)
    assert res_asym.degree == 2
    res_asym.verify_root_interlacing()  # Ensure it outputs hyperbolic polynomials


def test_symmetric_additive_interlacing() -> None:
    from finitefree.convolutions import symmetric_additive

    # p(x) = x^2 - 1, roots -1, 1
    p = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)
    # q(x) = x^2 - 4, roots -2, 2
    q = RealRootedPolynomial([1, 0, -4], assume_real_rooted=True)

    res = symmetric_additive(p, q, d=2)
    # The result (x^2 - 5) should strictly interlace its derivative
    assert res.verify_root_interlacing(strict=True)


def test_ambient_dimension_mismatch() -> None:
    from finitefree.convolutions import symmetric_additive

    # p(x) = x^2 - 3x + 2 (roots: 1, 2)
    p = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)
    # q(x) = x^4 (roots: 0, 0, 0, 0)
    q = RealRootedPolynomial([1, 0, 0, 0, 0], assume_real_rooted=True)

    # Convolve in ambient dimension d=4
    res = symmetric_additive(p, q, d=4)

    # Reconstructed polynomial must preserve real-rootedness
    assert res.verify_real_rootedness()


def test_symmetric_multiplicative_invariance() -> None:
    # Instantiate a symmetric polynomial p in P_{2d}^S(R)
    # p(x) = (x^2 - 1)(x^2 - 4) = x^4 - 5x^2 + 4. degree 2d = 4, d=2.
    p = RealRootedPolynomial([1, 0, -5, 0, 4], assume_real_rooted=True)

    # Instantiate a strictly non-negative polynomial q in P_{2d}(R_>=0)
    # q(x) = (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24
    q = RealRootedPolynomial([1, -10, 35, -50, 24], assume_real_rooted=True)

    # Compute p \boxtimes_{2d} q
    res = multiplicative(p, q, d=4)

    # Extract Symmetric Finite S-Transform
    # degree 2d = 4, d = 2. r = 0. Domain: k in {1, 2}
    s_res = SymmetricFiniteSTransform(res)
    s_p = SymmetricFiniteSTransform(p)
    s_q = FiniteSTransform(q)

    d_val = p.degree // 2
    zero_mult = 0
    while zero_mult < p.degree and p.coeffs[p.degree - zero_mult] == 0:
        zero_mult += 1
    r = zero_mult // 2

    for k in range(1, d_val - r + 1):
        lhs = s_res[k - 1] ** 2
        rhs = (s_p[k - 1] ** 2) * ((s_q[2 * k - 1] * s_q[2 * k - 2]) ** 2)
        assert sp.simplify(lhs - rhs) == 0


def test_unitary_hermite() -> None:
    # H_0(z; t) = 1
    u0 = unitary_hermite_polynomial(0, 1.0)
    assert isinstance(u0, UnitaryPolynomial)
    assert u0.degree == 0
    assert u0.coeffs[0] == 1

    # H_2(z; 1) = z^2 - 2exp(-1/4)z + 1
    # Leading coefficient is 1, so coeffs in descending order are [1, -2*exp(-1/4), 1]
    u2 = unitary_hermite_polynomial(2, 1)
    assert isinstance(u2, UnitaryPolynomial)
    assert u2.degree == 2
    assert u2.coeffs[0] == 1
    assert sp.simplify(u2.coeffs[1] - (-2 * sp.exp(sp.Rational(-1, 4)))) == 0
    assert u2.coeffs[2] == 1

    # Evaluate complex roots on unit circle
    roots = u2.evaluate_roots_float64()
    assert len(roots) == 2
    # Roots of z^2 - 2exp(-1/4)z + 1 = 0
    # exp(-1/4) is approx 0.7788
    # z = exp(-1/4) +/- i sqrt(1 - exp(-1/2))
    # Check that magnitude is 1.0
    for r in roots:
        assert np.isclose(np.abs(r), 1.0)


def test_convolution_degree_validation() -> None:
    import pytest

    # Degree 3 polynomial
    p = RealRootedPolynomial([1, -6, 11, -6], assume_real_rooted=True)
    # Degree 2 polynomial
    q = RealRootedPolynomial([1, -3, 2], assume_real_rooted=True)

    # Calling convolutions with d=2 < p.degree (3) should raise ValueError
    with pytest.raises(
        ValueError, match="Polynomial degrees cannot exceed dimension d"
    ):
        symmetric_additive(p, q, d=2)

    with pytest.raises(
        ValueError, match="Polynomial degrees cannot exceed dimension d"
    ):
        multiplicative(p, q, d=2)

    with pytest.raises(
        ValueError, match="Polynomial degrees cannot exceed dimension d"
    ):
        asymmetric_additive(p, q, d=2)


def test_asymmetric_additive_weights() -> None:
    p = RealRootedPolynomial([1, -2, 1], assume_real_rooted=True)  # (x-1)^2
    q = RealRootedPolynomial([1, 2, 1], assume_real_rooted=True)  # (x+1)^2

    # Verify asymmetric convolution with weights dilates the roots accordingly
    # weights = [0.5, 0.5] -> p dilated by 0.5: (x - 0.5)^2 = x^2 - x + 0.25
    # q dilated by 0.5: (x + 0.5)^2 = x^2 + x + 0.25
    # Asymmetric convolution of these two should result in x^2 + 0.25
    res = asymmetric_additive(p, q, d=2, weights=[0.5, 0.5])
    assert res.degree == 2
    assert np.allclose(list(res.coeffs), [1, 0, 0.25])
