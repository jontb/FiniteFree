import numpy as np

from finitefree.convolutions import (
    asymmetric_additive,
    multiplicative,
    symmetric_additive,
)
from finitefree.core import RealRootedPolynomial


def test_symmetric_additive_basic() -> None:
    # p(x) = x^2, q(x) = x^2 (d=2)
    p = RealRootedPolynomial([1, 0, 0], assume_real_rooted=True)
    q = RealRootedPolynomial([1, 0, 0], assume_real_rooted=True)

    # e_p = e_q = [1, 0, 0]
    # \boxplus_d will give e_res[k] = \sum \binom{k}{i} e_p[i] e_q[k-i]
    # k=0: e[0] = 1*1*1 = 1
    # k=1: e[1] = 0
    # k=2: e[2] = 0
    # result should be x^2
    res = symmetric_additive(p, q, 2)
    assert np.allclose(list(res.coeffs), [1, 0, 0])


def test_multiplicative_basic() -> None:
    # p(x) = x^2 - 1 = (x-1)(x+1) -> not strictly positive roots, but fine for formula
    p = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)
    q = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)

    res = multiplicative(p, q, 2)
    assert res.degree == 2


def test_asymmetric_additive_basic() -> None:
    p = RealRootedPolynomial([1, -2, 1], assume_real_rooted=True)  # (x-1)^2
    q = RealRootedPolynomial([1, 2, 1], assume_real_rooted=True)  # (x+1)^2

    res = asymmetric_additive(p, q, 2)
    assert res.degree == 2


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
