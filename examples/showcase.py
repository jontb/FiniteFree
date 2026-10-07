"""Executable tour of current exact coefficients and numerical root APIs.

Run from the checkout with PYTHONPATH=. python examples/showcase.py.
Assertions check the displayed identities; numerical roots are not interval
certificates. Every example uses explicit public factories and conventions.
"""

import numpy as np
import sympy as sp

from examples.multivariate_workflows import main as showcase_multivariate
from finitefree import (
    FiniteCauchyTransform,
    FiniteRTransform,
    FiniteSTransform,
    RealRootedPolynomial,
    hermite_polynomial,
    laguerre_polynomial,
)
from finitefree.convolutions import multiplicative, symmetric_additive


def build_symmetric_roots_poly(d: int) -> RealRootedPolynomial:
    """Construct roots at -1 and 1, with an extra zero for odd degree."""
    return RealRootedPolynomial.from_roots([-1, 1] * (d // 2) + [0] * (d % 2))


def build_positive_roots_poly(d: int) -> RealRootedPolynomial:
    """Construct the repeated positive root 2 exactly."""
    return RealRootedPolynomial.from_roots([2] * d)


def build_hermite_poly(d: int) -> RealRootedPolynomial:
    """Use the monic probabilist Hermite factory, retaining recurrence metadata."""
    return hermite_polynomial(d, physicist=False)


def build_laguerre_poly(d: int, c_ratio: float = 2.0) -> RealRootedPolynomial:
    """Construct monic L_d^(n-d)(d*x), with integer n=d*c_ratio >= d."""
    n = int(d * c_ratio)
    if d <= 0 or n < d or n != d * c_ratio:
        raise ValueError("Require positive d and integer n=d*c_ratio >= d")
    return laguerre_polynomial(d, n - d).dilation(sp.Rational(1, d))


def showcase_basics() -> None:
    """Check coefficient operations, cumulants and the multiplicative limit."""
    p = RealRootedPolynomial([1, -3, 2])
    assert p.verify_real_rootedness()
    assert list(p.normalized_coeffs()) == [1, sp.Rational(3, 2), 2]
    assert list(p.power(2).coeffs) == [1, -5, 4]

    q = RealRootedPolynomial.from_roots([-2, 2])
    assert list(symmetric_additive(p, q, 2).coeffs) == [1, -3, -2]
    repeated_one = RealRootedPolynomial.from_roots([1, 1])
    repeated_two = RealRootedPolynomial.from_roots([2, 2])
    assert list(multiplicative(repeated_one, repeated_two, 2).coeffs) == [1, -4, 4]
    z = sp.Symbol("z")
    assert sp.simplify(FiniteCauchyTransform(p) - (1 / (z - 1) + 1 / (z - 2)) / 2) == 0
    assert FiniteRTransform(p, order=3) == [sp.Rational(3, 2), sp.Rational(1, 2), 0]
    np.testing.assert_allclose(p.phi_d().evaluate_roots_float64(), [4 / 3, 1.5])
    print("Basic coefficient, cumulant and Phi identities passed.")


def showcase_asymptotics() -> None:
    """Show additive moments and exact S values for repeated positive roots."""
    for d in (10, 20, 40):
        p = build_symmetric_roots_poly(d)
        roots = symmetric_additive(p, p, d).evaluate_roots_float64(exact=False)
        np.testing.assert_allclose(np.var(roots), 2, atol=1e-12)
        values = FiniteSTransform(build_positive_roots_poly(d))
        assert all(v == sp.Rational(1, 2) for v in values)
        print(f"degree={d}: additive root variance={np.var(roots):.6f}; S(-1/2)=1/2")


def showcase_semicircle_mp() -> None:
    """Compare finite-degree Hermite moments and normalized Laguerre S values."""
    for d in (10, 20, 40):
        p = build_hermite_poly(d)
        roots = p.evaluate_roots_float64(exact=False)
        convolved = symmetric_additive(p, p, d).evaluate_roots_float64(exact=False)
        # The finite-degree variance is d-1, not the limiting approximation d.
        np.testing.assert_allclose(np.var(roots), d - 1, atol=1e-11)
        np.testing.assert_allclose(np.var(convolved), 2 * (d - 1), atol=1e-11)
        # Scaling He_d roots by sqrt(d) gives variance 1-1/d.
        np.testing.assert_allclose(np.var(roots / np.sqrt(d)), 1 - 1 / d)

        lag = build_laguerre_poly(d, c_ratio=2.0)
        value = FiniteSTransform(lag)[d // 2 - 1]
        expected = sp.Rational(d, 2 * d - d // 2 + 1)
        assert value == expected
        # For this normalization, S(-1/2) approaches 1/(2-1/2)=2/3.
        print(
            f"degree={d}: Hermite variance={np.var(roots):.6f}; Laguerre S(-1/2)={value}"
        )


if __name__ == "__main__":
    showcase_basics()
    showcase_asymptotics()
    showcase_semicircle_mp()
    showcase_multivariate()
