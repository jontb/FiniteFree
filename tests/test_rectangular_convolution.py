"""Rectangular additive convolution and rectangular cumulants.

Independent references: the unnormalized coefficient formula of Gribinski-Marcus
and Cuenca expanded with Python fractions (the implementation uses a scaled
Cauchy product of normalized coefficients); hand-expanded low-degree cases; the
closed form for Gaussian noise (expected minors of a non-central Wishart
matrix); and a seeded Haar Monte Carlo as numerical evidence only.
"""

import math
from fractions import Fraction as F
from typing import Any

import numpy as np
import pytest

from finitefree import (
    RealRootedPolynomial,
    inverse_rectangular_cumulants,
    matrix_rectangular_cumulants,
    rectangular_cumulants,
    sample_haar_orthogonal,
)
from finitefree.convolutions import asymmetric_additive
from tests._cases import case


def signed_coefficients(roots: list[F]) -> list[F]:
    """a_0..a_d with p(x) = sum_i (-1)^i a_i x^(d-i), by direct expansion."""
    e = [F(1)] + [F(0)] * len(roots)
    for r in roots:
        for k in range(len(roots), 0, -1):
            e[k] += r * e[k - 1]
    return e


def reference_convolution(a: list[F], b: list[F], d: int, m: int) -> list[F]:
    """Unnormalized (n, d) formula with n = m - d; returns descending coefficients."""
    f = math.factorial
    out = []
    for k in range(d + 1):
        total = F(0)
        for i in range(k + 1):
            j = k - i
            total += (
                F(f(d - i) * f(d - j), f(d) * f(d - k))
                * F(f(m - i) * f(m - j), f(m) * f(m - k))
                * a[i]
                * b[j]
            )
        out.append((-1) ** k * total)
    return out


def as_fractions(poly: RealRootedPolynomial) -> list[F]:
    return [F(int(c.p), int(c.q)) for c in poly.coeffs]


def fr(values: Any) -> list[F]:
    return [F(int(v.p), int(v.q)) for v in values]


ROOT_SETS = [
    ([F(2)], [F(5)]),
    ([F(1), F(3)], [F(2), F(7, 2)]),
    ([F(1, 2), F(2), F(5)], [F(1), F(3), F(7, 2)]),
    ([F(0), F(0), F(3), F(4)], [F(1, 3), F(1), F(2), F(9)]),
]


def test_rectangular_convolution_matches_unnormalized_formula() -> None:
    for a_roots, b_roots in ROOT_SETS:
        d = len(a_roots)
        p = RealRootedPolynomial.from_roots(a_roots)
        q = RealRootedPolynomial.from_roots(b_roots)
        a, b = signed_coefficients(a_roots), signed_coefficients(b_roots)
        for m in (d, d + 1, d + 3, 4 * d + 2):
            with case(d=d, m=m):
                got = as_fractions(asymmetric_additive(p, q, d, columns=m))
                assert got == reference_convolution(a, b, d, m)
        # The square case is the existing default, unchanged.
        assert as_fractions(asymmetric_additive(p, q, d)) == as_fractions(
            asymmetric_additive(p, q, d, columns=d)
        )


def test_rectangular_convolution_hand_expanded_cases() -> None:
    # d = 1: (sqrt(a) u + sqrt(b) v)^2 averages to a + b for every m.
    one = asymmetric_additive(
        RealRootedPolynomial.from_roots([2]),
        RealRootedPolynomial.from_roots([5]),
        1,
        columns=4,
    )
    assert as_fractions(one) == [1, -7]
    # d = 2, m = 3, roots (1, 3) and (2, 7/2):
    #   k = 1: a_1 + b_1 = 4 + 11/2 = 19/2
    #   k = 2: a_2 + b_2 + [1/2 * 2/3] a_1 b_1 = 3 + 7 + 22/3 = 52/3
    two = asymmetric_additive(
        RealRootedPolynomial.from_roots([1, 3]),
        RealRootedPolynomial.from_roots([2, F(7, 2)]),
        2,
        columns=3,
    )
    assert as_fractions(two) == [1, F(-19, 2), F(52, 3)]
    # Same inputs, square (m = 2): the cross weight is (1/2)^2, so 10 + 11/2.
    square = asymmetric_additive(
        RealRootedPolynomial.from_roots([1, 3]),
        RealRootedPolynomial.from_roots([2, F(7, 2)]),
        2,
    )
    assert as_fractions(square) == [1, F(-19, 2), F(31, 2)]


def test_rectangular_convolution_with_gaussian_noise_closed_form() -> None:
    # For C = S + sigma G (d x m, G iid standard normal),
    # E e_k(C C^T) = sum_i e_i(S S^T) sigma^(2(k-i)) (d-i)!(m-i)!/((d-k)!(m-k)!(k-i)!),
    # and the noise polynomial has e_j = sigma^(2j) j! C(d,j) C(m,j).
    f = math.factorial
    for d, m, sigma2, roots in (
        (2, 5, F(1, 7), [F(1, 2), F(3)]),
        (3, 4, F(2, 5), [F(0), F(1), F(4)]),
        (4, 9, F(1, 20), [F(1, 3), F(1), F(2), F(5)]),
    ):
        with case(d=d, m=m):
            noise_e = [
                sigma2**j * f(j) * math.comb(d, j) * math.comb(m, j)
                for j in range(d + 1)
            ]
            noise = RealRootedPolynomial(
                [(-1) ** j * noise_e[j] for j in range(d + 1)], assume_real_rooted=True
            )
            e = signed_coefficients(roots)
            expected = [
                (-1) ** k
                * sum(
                    e[i]
                    * sigma2 ** (k - i)
                    * F(f(d - i) * f(m - i), f(d - k) * f(m - k) * f(k - i))
                    for i in range(k + 1)
                )
                for k in range(d + 1)
            ]
            got = asymmetric_additive(
                RealRootedPolynomial.from_roots(roots), noise, d, columns=m
            )
            assert as_fractions(got) == expected


def test_rectangular_convolution_haar_monte_carlo_evidence() -> None:
    rng = np.random.default_rng(20261110)
    d, m, draws = 2, 4, 40000
    a, b = np.array([0.5, 2.0]), np.array([1.0, 3.0])
    A = np.zeros((d, m))
    B = np.zeros((d, m))
    A[np.arange(d), np.arange(d)] = np.sqrt(a)
    B[np.arange(d), np.arange(d)] = np.sqrt(b)
    total = np.zeros(d + 1)
    squares = np.zeros(d + 1)
    for _ in range(draws):
        C = A + sample_haar_orthogonal(d, rng=rng) @ B @ sample_haar_orthogonal(
            m, rng=rng
        )
        c = np.poly(np.linalg.eigvalsh(C @ C.T))
        total += c
        squares += c * c
    mean = total / draws
    se = np.sqrt(squares / draws - mean**2) / math.sqrt(draws)
    rect = asymmetric_additive(
        RealRootedPolynomial.from_roots([F(1, 2), 2]),
        RealRootedPolynomial.from_roots([1, 3]),
        d,
        columns=m,
    )
    square = asymmetric_additive(
        RealRootedPolynomial.from_roots([F(1, 2), 2]),
        RealRootedPolynomial.from_roots([1, 3]),
        d,
    )
    expected = np.array([float(c) for c in as_fractions(rect)])
    # Both coefficients agree; the determinant one separates m = 4 from m = 2.
    assert abs(mean[1] - expected[1]) < 5 * se[1]
    assert abs(mean[2] - expected[2]) < 5 * se[2]
    assert abs(mean[2] - float(as_fractions(square)[2])) > 20 * se[2]


def test_rectangular_cumulants_definition_additivity_and_dilation() -> None:
    # d = 1: K_2 = a_2 * 0!(m-1)!/(1! m!) = -root / m.
    assert fr(
        rectangular_cumulants(RealRootedPolynomial.from_roots([2]), columns=4)
    ) == [F(-1, 2)]
    # d = 2, m = 3, roots (1, 3): series 1 + c_1 y + c_2 y^2 with
    #   c_1 = -4 * (1! 2!)/(2! 3!) = -2/3,  c_2 = 3 * (0! 1!)/(2! 3!) = 1/4,
    #   K_2 = c_1 = -2/3,  K_4 = 2 (c_2 - c_1^2 / 2) = 1/2 - 4/9 = 1/18.
    got = rectangular_cumulants(RealRootedPolynomial.from_roots([1, 3]), columns=3)
    assert fr(got) == [F(-2, 3), F(1, 18)]
    for a_roots, b_roots in ROOT_SETS:
        d = len(a_roots)
        p = RealRootedPolynomial.from_roots(a_roots)
        q = RealRootedPolynomial.from_roots(b_roots)
        for m in (d, d + 2, 3 * d + 1):
            with case(d=d, m=m):
                kp = rectangular_cumulants(p, columns=m)
                kq = rectangular_cumulants(q, columns=m)
                kr = rectangular_cumulants(
                    asymmetric_additive(p, q, d, columns=m), columns=m
                )
                assert [x + y for x, y in zip(kp, kq)] == kr
                scaled = rectangular_cumulants(p.dilation(F(3, 2)), columns=m)
                assert fr(scaled) == [
                    F(3, 2) ** (i + 1) * k for i, k in enumerate(fr(kp))
                ]
                assert (
                    inverse_rectangular_cumulants(kp, columns=m)._fmpq_poly
                    == p._fmpq_poly
                )
                summed = inverse_rectangular_cumulants(
                    [x + y for x, y in zip(kp, kq)], columns=m, d=d
                )
                assert (
                    summed._fmpq_poly
                    == asymmetric_additive(p, q, d, columns=m)._fmpq_poly
                )


def test_rectangular_cumulants_ambient_dimension_and_samples() -> None:
    low = RealRootedPolynomial.from_roots([1, 3])
    embedded = RealRootedPolynomial.from_roots([0, 1, 3])
    assert rectangular_cumulants(low, columns=5, d=3) == rectangular_cumulants(
        embedded, columns=5
    )
    rng = np.random.default_rng(3)
    for d, m in ((1, 1), (2, 5), (4, 6)):
        with case(d=d, m=m):
            sample = rng.standard_normal((d, m))
            squared = np.linalg.svd(sample, compute_uv=False) ** 2
            exact = rectangular_cumulants(
                RealRootedPolynomial.from_roots(squared), columns=m
            )
            np.testing.assert_allclose(
                matrix_rectangular_cumulants(sample),
                [float(k.p) / float(k.q) for k in exact],
                rtol=1e-9,
                atol=1e-12,
            )
    complex_sample = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
    assert matrix_rectangular_cumulants(complex_sample).shape == (2,)


def test_rectangular_validation() -> None:
    p = RealRootedPolynomial.from_roots([1, 3])
    bad_columns: list[Any] = [1, True, 2.0, "3"]
    for columns in bad_columns:
        with case(columns=columns), pytest.raises(ValueError, match="columns"):
            asymmetric_additive(p, p, 2, columns=columns)
    for columns in (1, True):
        with case(columns=columns), pytest.raises(ValueError, match="columns"):
            rectangular_cumulants(p, columns=columns)
    with pytest.raises(ValueError, match="degree"):
        rectangular_cumulants(p, columns=4, d=1)
    with pytest.raises(ValueError, match="rational"):
        inverse_rectangular_cumulants([0.5, 0.25], columns=3)
    with pytest.raises(ValueError, match="exactly d"):
        inverse_rectangular_cumulants([F(1, 2)], columns=3, d=2)
    with pytest.raises(ValueError, match="at least one"):
        inverse_rectangular_cumulants([], columns=3)
    with pytest.raises(ValueError, match="columns"):
        matrix_rectangular_cumulants(np.ones((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        matrix_rectangular_cumulants(np.array([[1.0, np.inf]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        matrix_rectangular_cumulants(np.ones(3))
