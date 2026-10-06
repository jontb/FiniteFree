from typing import Any

import numpy as np
import pytest
import sympy as sp

from finitefree import RealRootedPolynomial, SymmetricFiniteSTransform


@pytest.mark.parametrize(
    "roots", [[-1, 1], [-2, -1, 1, 2], [-1, 0, 0, 1], [-2, -2, 2, 2]]
)
def test_standard_square_matches_independent_coefficient_ratios(
    roots: list[int],
) -> None:
    x = sp.Symbol("x")
    reference = sp.Poly(sp.prod(x - r for r in roots), x)
    n = len(roots)
    normalized = [
        (-1) ** k * reference.nth(n - k) / sp.binomial(n, k) for k in range(n + 1)
    ]
    count = (n - roots.count(0)) // 2
    expected = [normalized[2 * k - 2] / normalized[2 * k] for k in range(1, count + 1)]
    p = RealRootedPolynomial(reference.all_coeffs())
    legacy = SymmetricFiniteSTransform(p)
    explicit = SymmetricFiniteSTransform(p, convention="ratio")
    standard = SymmetricFiniteSTransform(p, convention="standard")
    assert legacy.tolist() == explicit.tolist() == expected
    assert [sp.simplify(s * s) for s in standard] == expected
    assert all(sp.im(s) > 0 and sp.re(s) == 0 for s in standard)
    numeric = SymmetricFiniteSTransform(p, exact=False, convention="standard")
    assert numeric.dtype == np.complex128
    np.testing.assert_allclose(numeric, [complex(v) for v in standard], rtol=2e-15)


def test_minimal_standard_and_legacy_example() -> None:
    p = RealRootedPolynomial([1, 0, -1])
    assert SymmetricFiniteSTransform(p).tolist() == [-1]
    assert SymmetricFiniteSTransform(p, convention="standard").tolist() == [sp.I]
    assert SymmetricFiniteSTransform(p, False).dtype == np.float64


@pytest.mark.parametrize("coefficients", [[1, 0, 1], [1, 0, -1, 0, -1]])
def test_standard_rejects_even_polynomials_with_nonreal_roots(
    coefficients: list[int],
) -> None:
    p = RealRootedPolynomial(coefficients)
    with pytest.raises(ValueError, match="real-rooted"):
        SymmetricFiniteSTransform(p, convention="standard")


def test_legacy_parity_only_behavior_is_preserved() -> None:
    p = RealRootedPolynomial([1, 0, 1])
    assert SymmetricFiniteSTransform(p).tolist() == [1]
    assert p._is_verified is False


@pytest.mark.parametrize("coefficients", [[1, -1], [1, -3, 2], [1]])
def test_standard_invalid_degree_or_symmetry(coefficients: list[int]) -> None:
    with pytest.raises(ValueError):
        SymmetricFiniteSTransform(
            RealRootedPolynomial(coefficients), convention="standard"
        )


@pytest.mark.parametrize("exact", [True, False])
def test_all_zero_roots_have_empty_node_domain(exact: bool) -> None:
    p = RealRootedPolynomial([1, 0, 0, 0, 0])
    result = SymmetricFiniteSTransform(p, exact=exact, convention="standard")
    assert result.size == 0
    assert result.dtype == (object if exact else np.complex128)


@pytest.mark.parametrize("scale", [sp.Rational(7, 3), -(10**500)])
def test_standard_retains_nonmonic_invariance(scale: Any) -> None:
    p = RealRootedPolynomial([scale, 0, -scale], monic=False)
    assert SymmetricFiniteSTransform(p, convention="standard").tolist() == [sp.I]


@pytest.mark.parametrize("root", [sp.Rational(10) ** 200, sp.Rational(10) ** -200])
def test_numeric_sqrt_precedes_float_narrowing(root: Any) -> None:
    p = RealRootedPolynomial([1, 0, -(root**2)], assume_real_rooted=True)
    result = SymmetricFiniteSTransform(p, exact=False, convention="standard")
    assert result[0].imag == pytest.approx(float(1 / root), rel=2e-15, abs=0)


@pytest.mark.parametrize("root", [sp.Rational(10) ** 400, sp.Rational(10) ** -400])
def test_numeric_standard_rejects_range_loss(root: Any) -> None:
    p = RealRootedPolynomial([1, 0, -(root**2)], assume_real_rooted=True)
    with pytest.raises(RuntimeError, match="complex128"):
        SymmetricFiniteSTransform(p, exact=False, convention="standard")


def test_symbolic_backend_and_unknown_conventions_fail_clearly() -> None:
    with pytest.raises(ValueError, match="rational"):
        SymmetricFiniteSTransform(
            RealRootedPolynomial([1, 0, -sp.sqrt(2)]), convention="standard"
        )
    with pytest.raises(ValueError, match="convention"):
        SymmetricFiniteSTransform(
            RealRootedPolynomial([1, 0, -1]),
            convention="unknown",  # type: ignore[arg-type]
        )
