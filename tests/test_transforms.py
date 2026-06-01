import numpy as np
import pytest
import sympy as sp

from finitefree.core import RealRootedPolynomial
from finitefree.transforms import (
    FiniteCauchyTransform,
    FiniteRTransform,
    FiniteSTransform,
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
    assert np.isclose(float(S[0]), (3 / 2) / 1)  # e_1 / e_0 = 1.5
    assert np.isclose(float(S[1]), 2 / (3 / 2))  # e_2 / e_1 = 4/3


def test_finite_s_transform_positivity_violation() -> None:
    # p(x) = x^2 - 1 (roots 1, -1)
    # c_0 = 1, c_1 = 0, c_2 = -1
    # e_1 = 0
    p = RealRootedPolynomial([1, 0, -1], assume_real_rooted=True)
    with pytest.raises(ValueError, match="Strict positivity constraint violated"):
        FiniteSTransform(p)


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
