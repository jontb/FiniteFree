import itertools
import math
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    DiscreteFiniteKernel,
    DPPCountDistribution,
    OrthogonalPolynomialKernel,
    RealRootedPolynomial,
    count_distribution_continuous,
    count_distribution_discrete,
    gap_probability_continuous,
    gap_probability_discrete,
    legendre_polynomial,
)
from tests._cases import case
from tests.test_kernel_evaluation import hermite_kernel


def test_prepared_batches_match_exact_basis_and_own_normalization() -> None:
    hermite = hermite_kernel(7)
    # Signed, enormous basis scales must cancel against squared norms.
    scales = [(-1) ** j * flint.fmpq(10) ** (200 * j) for j in range(8)]
    scaled = OrthogonalPolynomialKernel(
        [
            RealRootedPolynomial(p._fmpq_poly * scales[j], monic=False)
            for j, p in enumerate(hermite.polys)
        ],
        [h * scales[j] ** 2 * 3 for j, h in enumerate(hermite.norms)],
    )
    legendre = [legendre_polynomial(j) for j in range(8)]
    ratios = [
        sp.Rational(p.coeffs[0]) / sp.LC(sp.legendre(j, sp.Symbol("x")), sp.Symbol("x"))
        for j, p in enumerate(legendre)
    ]
    generic = OrthogonalPolynomialKernel(
        legendre, [2 * ratios[j] ** 2 / (2 * j + 1) for j in range(7)]
    )
    shifted = OrthogonalPolynomialKernel(
        [p.dilation(2).shift(3) for p in hermite.polys],
        [5 * h * 4**j for j, h in enumerate(hermite.norms)],
    )
    for kernel in [hermite, scaled, shifted, generic, hermite_kernel(0)]:
        with case(rank=kernel.n, norms=kernel.norms):
            prepared = kernel.prepare_numeric()
            xs = np.array([-0.75, 9, 0.25, 9, 0.25 + 1e-12])[::2]
            ys = np.array([0.1, 0.3])
            expected = [
                [
                    float(
                        sum(
                            p.evaluate(sp.Rational(float(x)))
                            * p.evaluate(sp.Rational(float(y)))
                            / h
                            for p, h in zip(kernel.polys, kernel.norms)
                        )
                    )
                    for y in ys
                ]
                for x in xs
            ]
            np.testing.assert_allclose(
                prepared.matrix(xs, ys), expected, rtol=2e-12, atol=1e-14
            )
            assert prepared.matrix([], ys).shape == (0, 2)
            assert prepared.matrix(xs, []).shape == (3, 0)
            assert prepared.basis(xs).shape == (3, kernel.n)
            first = prepared.matrix(xs)
            leaked = prepared.basis(xs)
            leaked[:] = 999
            kernel.polys[0]._fmpq_poly[0] = 888
            np.testing.assert_array_equal(prepared.matrix(xs), first)
            # Preparation cannot change the exact scalar path.
            assert kernel(0, 1) == sum(
                p.evaluate(0) * p.evaluate(1) / h
                for p, h in zip(kernel.polys, kernel.norms)
            )


def test_generic_normalization_precedes_float_conversion() -> None:
    scale = sp.Integer(10) ** 400
    # Rank one, non-Hermite p1 triggers the generic path.
    kernel = OrthogonalPolynomialKernel(
        [RealRootedPolynomial([scale], monic=False), RealRootedPolynomial([1, -2])],
        [scale**2 * 2],
        leading_coeffs=[scale, 2],
    )
    np.testing.assert_allclose(
        kernel.prepare_numeric().matrix([0, 1]), np.full((2, 2), 0.5)
    )


def test_quadrature_matches_analytic_rank_two_restriction_and_reuses_density() -> None:
    # Orthonormal 1/sqrt(2), sqrt(3/2)*x in Lebesgue measure on [-1,1].
    kernel = OrthogonalPolynomialKernel(
        [
            RealRootedPolynomial([1]),
            RealRootedPolynomial([1, 0]),
            RealRootedPolynomial([1, 0, sp.Rational(-1, 3)]),
        ],
        [2, sp.Rational(2, 3)],
    )
    called = []

    def density(x: float) -> float:
        called.append(x)
        return 1.0

    quadrature = kernel.prepare_numeric().quadrature(0, 1, 8, density)
    gram = np.array([[0.5, math.sqrt(3) / 4], [math.sqrt(3) / 4, 0.5]])
    np.testing.assert_allclose(quadrature.gram, gram, atol=5e-15)
    law = quadrature.count_distribution()
    # det(G)=1/16, trace(G)=1 => PMF=(1/16,7/8,1/16).
    np.testing.assert_allclose(law.pmf(), [1 / 16, 7 / 8, 1 / 16], atol=2e-15)
    assert law.mean == pytest.approx(1)
    assert law.variance == pytest.approx(1 / 8)
    assert quadrature.gap_probability() == pytest.approx(1 / 16)
    assert law.estimated
    assert len(called) == 8
    for array in [
        quadrature.nodes,
        quadrature.weights,
        quadrature.gram,
        law.eigenvalues,
        law.pmf(),
    ]:
        array[:] = 0
    np.testing.assert_allclose(quadrature.gram, gram, atol=5e-15)
    assert len(called) == 8
    assert law.pmf(0) == pytest.approx(gap_probability_continuous(kernel, 0, 1, 8))
    np.testing.assert_allclose(
        count_distribution_continuous(kernel, 0, 1, 8).pmf(), law.pmf()
    )
    empty = hermite_kernel(0).prepare_numeric().quadrature(-1, 1)
    np.testing.assert_array_equal(empty.count_distribution().pmf(), [1])


def test_count_law_matches_enumeration_rotation_and_exact_gap() -> None:
    probabilities = np.array([0, 0.13, 0.41, 0.87, 1])
    reference = np.zeros(6)
    for bits in itertools.product([0, 1], repeat=5):
        reference[sum(bits)] += math.prod(
            p if bit else 1 - p for p, bit in zip(probabilities, bits)
        )
    rotation, _ = np.linalg.qr(np.random.default_rng(12).normal(size=(5, 5)))
    law = DPPCountDistribution((rotation * probabilities) @ rotation.T)
    np.testing.assert_allclose(law.pmf(), reference, atol=2e-15)
    assert law.mean == pytest.approx(sum(probabilities))
    assert law.variance == pytest.approx(sum(probabilities * (1 - probabilities)))
    for k in range(-2, 8):
        assert law.pmf(k) == pytest.approx(reference[k] if 0 <= k < 6 else 0)
        assert law.cdf(k) == pytest.approx(sum(reference[: k + 1]) if k >= 0 else 0)
        assert law.sf(k) == pytest.approx(sum(reference[k + 1 :]) if k >= 0 else 1)
    exact = DiscreteFiniteKernel(
        [[sp.Rational(1, 2), sp.Rational(1, 4)], [sp.Rational(1, 4), sp.Rational(1, 2)]]
    )
    assert count_distribution_discrete(exact, [0, 1]).pmf(0) == pytest.approx(
        float(gap_probability_discrete(exact, [0, 1]))
    )
    np.testing.assert_array_equal(count_distribution_discrete(exact, []).pmf(), [1])
    with pytest.raises(ValueError, match="distinct"):
        count_distribution_discrete(exact, [0, 0])
    rare = DPPCountDistribution(np.diag([1e-20, 1e-20]))
    assert rare.cdf(1) == 1
    assert rare.sf(1) == pytest.approx(1e-40, rel=1e-14, abs=0)
    # This distribution previously rounded cdf(31) to 1.0000000000000002.
    # Keep the tiny upper tail even when the corresponding CDF rounds to 1.
    rng = np.random.default_rng(2026)
    rng.random(11000)
    probabilities = rng.random(32 * 282)[-32:]
    rounded = DPPCountDistribution(np.diag(probabilities))
    assert 0 <= rounded.cdf(31) <= 1
    assert rounded.sf(31) == pytest.approx(math.prod(probabilities), rel=2e-14, abs=0)
    for k in range(-1, 33):
        assert 0 <= rounded.cdf(k) <= 1
        assert 0 <= rounded.sf(k) <= 1


def test_numeric_validation_and_roundoff_diagnostics() -> None:
    value: Any
    for value in [
        np.zeros((2, 3)),
        [1, 2],
        [[1, 1], [0, 1]],
        [[1.001]],
        [[-0.001]],
        [[np.inf]],
        [[1 + 0j]],
        [[sp.I]],
    ]:
        with case(value=value), pytest.raises(ValueError):
            DPPCountDistribution(value)
    law = DPPCountDistribution(np.diag([-5e-13, 1 + 5e-13]))
    assert law.spectral_correction == pytest.approx(5e-13, abs=1e-16)
    np.testing.assert_array_equal(law.pmf(), [0, 1, 0])
    prepared = hermite_kernel(5).prepare_numeric()
    for value in [[np.nan], [np.inf], [1 + 0j], [[1]], [sp.I]]:
        with case(value=value), pytest.raises(ValueError):
            prepared.matrix(value)
    with pytest.raises(RuntimeError, match="range"):
        prepared.matrix([1e200])
    order: Any
    density: Any
    for a, b, order, density in [
        (1, 0, 3, None),
        (0, np.inf, 3, None),
        (0, 1, 0, None),
        (0, 1, 2.2, None),
        (0, 1, True, None),
        (0, 1, 3, lambda x: -1),
        (0, 1, 3, lambda x: np.nan),
        (0, 1, 3, lambda x: 1j),
    ]:
        with case(a=a, b=b, order=order), pytest.raises(ValueError):
            prepared.quadrature(a, b, order, density)
    with pytest.raises(ValueError, match="eigenvalues"):
        prepared.quadrature(-3, 3, 20).count_distribution()  # Missing Gaussian measure.
    for norm in [0, -1, sp.Symbol("h"), sp.I]:
        with case(norm=norm), pytest.raises(ValueError, match="Basis"):
            OrthogonalPolynomialKernel(
                [RealRootedPolynomial([1]), RealRootedPolynomial([1, 0])], [norm]
            ).prepare_numeric()
