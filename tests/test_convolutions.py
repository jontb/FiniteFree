from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest
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
from tests._cases import case


@dataclass(frozen=True)
class _Convolution:
    operation: Callable[..., RealRootedPolynomial]
    p: list[int]
    q: list[int]
    d: int
    expected: list[Any]
    weights: list[float] | None = None
    real_rooted: bool | None = None
    interlacing: bool | None = None  # strict flag; None skips the check


def test_low_degree_convolutions_against_hand_expansions() -> None:
    # Write p = sum_i (-1)^i a_i x^(d-i) and q likewise with b_j. The coefficient
    # of (-1)^k x^(d-k) is sum_(i+j=k) w a_i b_j, where the symmetric weight is
    # w = (d-i)!(d-j)!/(d!(d-k)!) and the asymmetric weight is w**2; the
    # multiplicative coefficient is a_k b_k / binomial(d, k). Weights dilate the
    # asymmetric inputs first. Each expected value is expanded from these sums.
    cases = {
        "symmetric roots (1,2) and (1,3)": _Convolution(
            symmetric_additive, [1, -3, 2], [1, -4, 3], 2, [1, -7, 11]
        ),
        "symmetric roots (-1,1) and (-2,2)": _Convolution(
            symmetric_additive, [1, 0, -1], [1, 0, -4], 2, [1, 0, -5], interlacing=True
        ),
        # x^4 is the identity at d=4, so p acquires two ambient zero roots.
        "symmetric ambient identity": _Convolution(
            symmetric_additive,
            [1, -3, 2],
            [1, 0, 0, 0, 0],
            4,
            [1, -3, 2, 0, 0],
            real_rooted=True,
        ),
        # Real-rooted inputs with mixed signs give the formal result x^2 + 1.
        "multiplicative x^2 - 1": _Convolution(
            multiplicative, [1, 0, -1], [1, 0, -1], 2, [1, 0, 1], real_rooted=False
        ),
        "asymmetric (x-1)^2 and (x+1)^2": _Convolution(
            asymmetric_additive, [1, -2, 1], [1, 2, 1], 2, [1, 0, 1]
        ),
        "asymmetric dilated by 1/2": _Convolution(
            asymmetric_additive,
            [1, -2, 1],
            [1, 2, 1],
            2,
            [1, 0, sp.Rational(1, 4)],
            weights=[0.5, 0.5],
        ),
        "asymmetric roots (2,3) and (2,-1)": _Convolution(
            asymmetric_additive,
            [1, -5, 6],
            [1, -1, -2],
            2,
            [1, -6, sp.Rational(21, 4)],
            interlacing=False,
        ),
    }
    for name, c in cases.items():
        with case(convolution=name):
            p = RealRootedPolynomial(c.p, assume_real_rooted=True)
            q = RealRootedPolynomial(c.q, assume_real_rooted=True)
            kwargs = {} if c.weights is None else {"weights": c.weights}
            res = c.operation(p, q, c.d, **kwargs)
            assert res.degree == len(c.expected) - 1
            assert list(res.coeffs) == c.expected
            if c.real_rooted is True:
                assert res.verify_real_rootedness()
            elif c.real_rooted is False:
                with pytest.raises(ValueError, match="not real-rooted"):
                    res.verify_real_rootedness()
            if c.interlacing is not None:
                assert res.verify_root_interlacing(strict=c.interlacing)


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
