"""Prepared bases of every family with an exact recurrence, at ranks >= 30.

References use only the supplied polynomials and closed-form norms: exact
rational evaluation divided by a 40-digit square root, and exact rational
interval Gram matrices whose characteristic polynomial gives the count law.
"""

import math
from decimal import Decimal, localcontext
from typing import Any, Callable, Optional

import flint
import numpy as np
import pytest

from finitefree import (
    OrthogonalPolynomialKernel,
    RealRootedPolynomial,
    jacobi_polynomial,
    laguerre_polynomial,
    legendre_polynomial,
)
from tests._cases import case


def monic_jacobi_norm(j: int, alpha: int, beta: int) -> Any:
    """Closed-form monic Jacobi norm, integral of q_j**2 (1-x)**alpha (1+x)**beta."""

    def gamma(m: int) -> int:
        return math.factorial(m - 1)

    s = alpha + beta
    numerator = (
        2 ** (2 * j + s + 1)
        * math.factorial(j)
        * gamma(j + alpha + 1)
        * gamma(j + beta + 1)
        * gamma(j + s + 1)
    )
    return flint.fmpq(numerator, gamma(2 * j + s + 1) * gamma(2 * j + s + 2))


def monic_laguerre_norm(j: int, alpha: Any) -> Any:
    """j! (alpha+1)_j for the probability measure x**alpha exp(-x)/Gamma(alpha+1)."""
    result = flint.fmpq(math.factorial(j))
    for k in range(1, j + 1):
        result *= alpha + k
    return result


def build(
    monic: list[Any], monic_norms: list[Any], scales: Optional[list[Any]] = None
) -> OrthogonalPolynomialKernel:
    """Kernel from monic data, with optional signed basis rescalings s_j."""
    scales = scales or [flint.fmpq(1)] * len(monic)
    return OrthogonalPolynomialKernel(
        [
            RealRootedPolynomial(q * s, monic=False, assume_real_rooted=True)
            for q, s in zip(monic, scales)
        ],
        [h * s**2 for h, s in zip(monic_norms, scales)],
    )


def exact(x: float) -> Any:
    return flint.fmpq(*float(x).as_integer_ratio())


def reference_basis(kernel: OrthogonalPolynomialKernel, xs: Any) -> Any:
    """p_j(x)/sqrt(h_j) from exact rational evaluation and a 40-digit root."""
    polys, norms = kernel.polys, kernel.norms
    result = np.empty((len(xs), kernel.n))
    with localcontext() as context:
        context.prec = 40
        for j, (p, h) in enumerate(zip(polys, norms)):
            root = (Decimal(int(h.p)) / Decimal(int(h.q))).sqrt()
            for i, x in enumerate(xs):
                value = p._fmpq_poly(exact(x))
                result[i, j] = float(
                    Decimal(int(value.p)) / Decimal(int(value.q)) / root
                )
    return result


def exact_count_law(
    monic: list[Any], monic_norms: list[Any], density: Any, a: float, b: float
) -> tuple[list[Any], Any, Any]:
    """Exact PMF, mean and variance on [a,b] for a polynomial density.

    A = diag(h)^-1 M, with M_ij the exact integral of q_i q_j density, is similar
    to the restricted Gram matrix (and invariant under basis rescaling). With
    charpoly(A)(s) = sum c_k s**k, sum_k P(N=k) z**k = sum_k c_k (1-z)**(n-k).
    """
    n = len(monic_norms)
    lower, upper = exact(a), exact(b)
    entries = []
    for i in range(n):
        for j in range(n):
            antiderivative = (monic[i] * monic[j] * density).integral()
            entries.append(
                (antiderivative(upper) - antiderivative(lower)) / monic_norms[i]
            )
    matrix = flint.fmpq_mat(n, n, entries)
    characteristic = matrix.charpoly()
    complement = flint.fmpq_poly([1, -1])
    generating = sum(
        (characteristic[k] * complement ** (n - k) for k in range(n + 1)),
        flint.fmpq_poly([]),
    )
    trace = sum((matrix[i, i] for i in range(n)), flint.fmpq(0))
    square = matrix * matrix
    variance = trace - sum((square[i, i] for i in range(n)), flint.fmpq(0))
    return [generating[k] for k in range(n + 1)], trace, variance


def as_float(values: Any) -> Any:
    with localcontext() as context:
        context.prec = 40
        return np.array([float(Decimal(int(v.p)) / Decimal(int(v.q))) for v in values])


def legendre_data(n: int) -> tuple[list[Any], list[Any]]:
    monic = [legendre_polynomial(j)._fmpq_poly for j in range(n + 1)]
    return monic, [monic_jacobi_norm(j, 0, 0) for j in range(n)]


def jacobi_data(n: int) -> tuple[list[Any], list[Any]]:
    monic = [jacobi_polynomial(j, 2, 1)._fmpq_poly for j in range(n + 1)]
    return monic, [monic_jacobi_norm(j, 2, 1) for j in range(n)]


def signed_scales(n: int) -> list[Any]:
    # Signs (+,-,-,+,...) and growing rational magnitudes.
    return [
        (-1) ** (j * (j + 1) // 2) * flint.fmpq(j + 2, 3) * flint.fmpq(10) ** (5 * j)
        for j in range(n + 1)
    ]


def affine(
    monic: list[Any], monic_norms: list[Any], shift: Any, dilation: Any
) -> tuple[list[Any], list[Any]]:
    """Monic c**j q_j((x-s)/c); Lebesgue mass and norms scale by c**(2j+1)."""
    substitution = flint.fmpq_poly([-shift / dilation, 1 / dilation])
    return (
        [q(substitution) * dilation**j for j, q in enumerate(monic)],
        [h * dilation ** (2 * j + 1) for j, h in enumerate(monic_norms)],
    )


def test_prepared_basis_matches_exact_rational_evaluation_at_high_rank() -> None:
    laguerre_alpha = flint.fmpq(1, 2)
    laguerre = (
        [laguerre_polynomial(j, "1/2")._fmpq_poly for j in range(31)],
        [monic_laguerre_norm(j, laguerre_alpha) for j in range(30)],
    )
    shifted = affine(*legendre_data(30), flint.fmpq(3), flint.fmpq(2))
    families: list[tuple[str, tuple[list[Any], list[Any]], Any, Any]] = [
        ("Legendre", legendre_data(50), np.linspace(-1, 1, 41), None),
        ("signed Legendre", legendre_data(40), np.linspace(-1, 1, 23), True),
        ("Jacobi(2,1)", jacobi_data(40), np.linspace(-1, 1, 23), True),
        ("Laguerre(1/2)", laguerre, np.linspace(0, 110, 23), True),
        ("shifted Legendre", shifted, np.linspace(1, 5, 17), True),
    ]
    for name, (monic, norms), xs, signed in families:
        with case(family=name, rank=len(norms)):
            scales = signed_scales(len(monic)) if signed else None
            kernel = build(monic, norms, scales)
            expected = reference_basis(kernel, xs)
            prepared = kernel.prepare_numeric()
            basis = prepared.basis(xs)
            # Orthonormal values grow away from the bulk; compare per column.
            scale = np.maximum(1, np.max(np.abs(expected), axis=0))
            assert np.max(np.abs(basis - expected) / scale) < 5e-14
            # Every column keeps the sign of its supplied leading coefficient.
            # Monic polynomials are positive beyond their largest zero.
            far = np.array([1e3 + float(xs[-1])])
            signs = [1] * kernel.n
            if scales is not None:
                signs = [1 if s > 0 else -1 for s in scales[: kernel.n]]
            np.testing.assert_array_equal(np.sign(prepared.basis(far)[0]), signs)
            # Basis sums agree with the exact rational kernel at a few points.
            for i, k in [(1, -2), (len(xs) // 2, len(xs) // 2)]:
                x, y = xs[i], xs[k]
                magnitude = float(np.abs(expected[i]) @ np.abs(expected[k]))
                assert prepared.matrix([x], [y])[0, 0] == pytest.approx(
                    float(kernel(exact(x), exact(y))), rel=0, abs=1e-13 * magnitude
                )


def test_prepared_count_laws_match_exact_rational_restrictions() -> None:
    lebesgue = flint.fmpq_poly([1])
    jacobi_density = flint.fmpq_poly([1, -1]) ** 2 * flint.fmpq_poly([1, 1])

    def jacobi_weight(x: float) -> float:
        return (1 - x) ** 2 * (1 + x)

    scenarios: list[
        tuple[str, tuple[list[Any], list[Any]], Any, Any, float, float, Any]
    ] = [
        ("Legendre", legendre_data(30), lebesgue, None, -1.0, 1.0, False),
        ("Legendre", legendre_data(30), lebesgue, None, 0.75, 1.0, True),
        ("Legendre", legendre_data(50), lebesgue, None, -1.0, 1.0, True),
        ("Legendre", legendre_data(50), lebesgue, None, 0.0, 1.0, True),
        ("Legendre", legendre_data(40), lebesgue, None, 0.75, 1.0, False),
        ("Jacobi(2,1)", jacobi_data(30), jacobi_density, jacobi_weight, -1, 1, True),
        ("Jacobi(2,1)", jacobi_data(30), jacobi_density, jacobi_weight, 0.5, 1, False),
        (
            "shifted Legendre",
            affine(*legendre_data(30), flint.fmpq(3), flint.fmpq(2)),
            lebesgue,
            None,
            4.5,
            5.0,
            True,
        ),
    ]
    for name, (monic, norms), density, weight, a, b, signed in scenarios:
        n = len(norms)
        with case(family=name, rank=n, interval=(a, b)):
            pmf, mean, variance = exact_count_law(monic[:n], norms, density, a, b)
            kernel = build(monic, norms, signed_scales(len(monic)) if signed else None)
            # n + 2 Gauss nodes integrate these degree <= 2n+1 integrands exactly.
            restriction = kernel.prepare_numeric().quadrature(a, b, n + 2, weight)
            # Basis evaluation alone: the same float nodes and weights with the
            # exactly evaluated basis (errors were O(1) before the recurrence).
            nodes = reference_basis(kernel, restriction.nodes)
            weighted = nodes * restriction.weights[:, None]
            np.testing.assert_allclose(
                restriction.gram, weighted.T @ nodes, rtol=0, atol=2e-14
            )
            if pmf[-1] == 1:
                # Full support: the exact restriction is the identity, so every
                # eigenvalue sits on the count law's documented 1 + 1e-12
                # validation boundary. Roundoff there is platform-dependent
                # (macOS exceeded it at rank 50, Linux at rank 30 came within
                # 6e-13), so check the Gram matrix rather than the law.
                np.testing.assert_allclose(
                    restriction.gram, np.eye(n), rtol=0, atol=1e-11
                )
                continue
            # Whole estimate, including float64 nodes/weights and eigensolver
            # roundoff.
            law = restriction.count_distribution()
            np.testing.assert_allclose(law.pmf(), as_float(pmf), rtol=0, atol=2e-11)
            assert law.mean == pytest.approx(float(mean), rel=0, abs=2e-11)
            assert law.variance == pytest.approx(float(variance), rel=0, abs=2e-11)
            assert restriction.gap_probability() == pytest.approx(
                float(pmf[0]), rel=0, abs=2e-11
            )
            assert law.estimated
            assert law.spectral_correction < 1e-12


def test_signed_verified_hermite_basis_keeps_column_signs() -> None:
    # He basis with p1 = -x, standard Gaussian measure of mass one.
    signed = [
        RealRootedPolynomial([1]),
        RealRootedPolynomial([-1, 0], monic=False),
        RealRootedPolynomial([1, 0, -1]),
    ]
    kernel = OrthogonalPolynomialKernel(signed, [1, 1])
    assert kernel._hermite_parameters is not None
    xs = [2.0, -0.5, 0.0]
    expected = reference_basis(kernel, xs)
    np.testing.assert_array_equal(expected[:, 1], [-2, 0.5, 0])
    np.testing.assert_array_equal(kernel.prepare_numeric().basis(xs), expected)
    # A scaled, shifted Hermite family with alternating leading signs.
    x = flint.fmpq_poly([flint.fmpq(-2, 3), 1])  # x - 2/3
    variance, mass = flint.fmpq(1, 7), flint.fmpq(5, 4)
    monic, norms = [flint.fmpq_poly([1])], []
    previous = flint.fmpq_poly([])
    for j in range(40):
        norms.append(mass * math.factorial(j) * variance**j)
        previous, current = monic[-1], x * monic[-1] - j * variance * previous
        monic.append(current)
    kernel = build(monic, norms, signed_scales(len(monic)))
    assert kernel._hermite_parameters is not None
    points = np.linspace(-0.5, 2, 11)
    expected = reference_basis(kernel, points)
    scale = np.maximum(1, np.max(np.abs(expected), axis=0))
    basis = kernel.prepare_numeric().basis(points)
    assert np.max(np.abs(basis - expected) / scale) < 5e-14


def test_unverified_bases_keep_coefficient_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original: Callable[..., Any] = np.polyval

    def counted(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(np, "polyval", counted)
    monic, norms = legendre_data(6)
    inconsistent = norms.copy()
    inconsistent[3] *= 2
    monomials = [flint.fmpq_poly([0] * j + [1]) for j in range(7)]
    # q_4 + q_1/5 keeps every leading and subleading recurrence coefficient and
    # norm ratio; only the exact three-term identity detects the extra term.
    perturbed = monic.copy()
    perturbed[4] += monic[1] / 5
    # No basis satisfies a recurrence consistent with its norms, yet each
    # defines a valid finite basis sum that the prepared kernel must reproduce.
    for name, kernel in [
        ("inconsistent norms", build(monic, inconsistent)),
        ("monomials", build(monomials, [flint.fmpq(1, j + 1) for j in range(6)])),
        ("lower-order perturbation", build(perturbed, norms)),
    ]:
        with case(basis=name):
            before = calls
            xs = np.linspace(-1, 1, 7)
            basis = kernel.prepare_numeric().basis(xs)
            assert calls == before + kernel.n
            np.testing.assert_allclose(
                basis, reference_basis(kernel, xs), rtol=1e-14, atol=1e-15
            )
    verified = build(*legendre_data(50))
    before = calls
    assert np.all(np.isfinite(verified.prepare_numeric().basis([0.3, 1.0])))
    assert calls == before  # The verified recurrence avoids monomial coefficients.
