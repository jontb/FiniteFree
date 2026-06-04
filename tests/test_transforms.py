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
