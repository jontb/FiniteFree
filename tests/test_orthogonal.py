from typing import Any, Callable

import pytest
import sympy as sp

from finitefree import (
    chebyshev_t_polynomial,
    chebyshev_u_polynomial,
    hahn_polynomial,
    hermite_polynomial,
    jack_polynomial,
    jacobi_polynomial,
    krawtchouk_polynomial,
    laguerre_polynomial,
    legendre_polynomial,
)
from tests._cases import case


def test_univariate_families_match_monic_closed_forms() -> None:
    half = sp.Rational(1, 2)
    families: dict[str, tuple[Callable[[], Any], list[Any]]] = {
        "Jacobi degree 0": (lambda: jacobi_polynomial(0, 1, 1), [1]),
        # P_1^(1,2)(x) = -1/2 + 5x/2, monic x - 1/5.
        "Jacobi P_1^(1,2)": (
            lambda: jacobi_polynomial(1, 1, 2),
            [1, sp.Rational(-1, 5)],
        ),
        # P_2^(0,0)(x) = (3x^2 - 1)/2, monic x^2 - 1/3.
        "Jacobi P_2^(0,0)": (
            lambda: jacobi_polynomial(2, 0, 0),
            [1, 0, sp.Rational(-1, 3)],
        ),
        "Hahn degree 0": (lambda: hahn_polynomial(0, 1, 1, 5), [1]),
        # Q_1(x; a, b, N) = 1 - (a + b + 2) x / ((a + 1) N) = 1 - x/2, monic x - 2.
        "Hahn Q_1(1, 1, 4)": (lambda: hahn_polynomial(1, 1, 1, 4), [1, -2]),
        # Physicist H_2(x) = 4x^2 - 2, monic x^2 - 1/2.
        "physicist Hermite H_2": (
            lambda: hermite_polynomial(2, physicist=True),
            [1, 0, -half],
        ),
        # Probabilist He_3(x) = x^3 - 3x is already monic.
        "probabilist Hermite He_3": (
            lambda: hermite_polynomial(3, physicist=False),
            [1, 0, -3, 0],
        ),
        # L_2^(1)(x) = x^2/2 - 3x + 3, monic x^2 - 6x + 6.
        "Laguerre L_2^(1)": (lambda: laguerre_polynomial(2, 1), [1, -6, 6]),
        # K_2(x; 1/2, 4) = 2F1(-2, -x; -4; 2) = x^2/3 - 4x/3 + 1, monic x^2 - 4x + 3.
        "Krawtchouk K_2(1/2, 4)": (
            lambda: krawtchouk_polynomial(2, half, 4),
            [1, -4, 3],
        ),
        # T_2(x) = 2x^2 - 1 and U_2(x) = 4x^2 - 1.
        "Chebyshev T_2": (lambda: chebyshev_t_polynomial(2), [1, 0, -half]),
        "Chebyshev U_2": (
            lambda: chebyshev_u_polynomial(2),
            [1, 0, sp.Rational(-1, 4)],
        ),
        # P_3(x) = (5x^3 - 3x)/2, monic x^3 - 3x/5.
        "Legendre P_3": (
            lambda: legendre_polynomial(3),
            [1, 0, sp.Rational(-3, 5), 0],
        ),
    }
    for name, (construct, expected) in families.items():
        with case(family=name):
            p = construct()
            assert p.degree == len(expected) - 1
            assert list(p.coeffs) == expected


def test_univariate_families_reject_invalid_parameters() -> None:
    invalid: dict[str, Callable[[], Any]] = {
        "Laguerre negative degree": lambda: laguerre_polynomial(-1, 1),
        "Krawtchouk negative degree": lambda: krawtchouk_polynomial(-1, 0.5, 4),
        "Krawtchouk degree above N": lambda: krawtchouk_polynomial(5, 0.5, 4),
        "Krawtchouk zero probability": lambda: krawtchouk_polynomial(2, 0.0, 4),
        "Chebyshev T negative degree": lambda: chebyshev_t_polynomial(-1),
        "Chebyshev U negative degree": lambda: chebyshev_u_polynomial(-1),
        "Legendre negative degree": lambda: legendre_polynomial(-1),
    }
    for name, construct in invalid.items():
        with case(parameters=name):
            with pytest.raises(ValueError):
                construct()


def test_jack_polynomials() -> None:
    # m = 3 variables, partition [1], alpha = 2
    # J_[1]^(2)(x1, x2, x3) = x1 + x2 + x3
    p1 = jack_polynomial(3, [1], 2)
    x1, x2, x3 = p1.variables
    assert sp.simplify(p1.expr - (x1 + x2 + x3)) == 0

    # m = 2 variables, partition [2], alpha = 3
    # J_[2]^(3)(x1, x2) = 4 * (x1^2 + x2^2) + 2 * x1 * x2
    p2 = jack_polynomial(2, [2], 3)
    vx1, vx2 = p2.variables
    expected = 4 * (vx1**2 + vx2**2) + 2 * vx1 * vx2
    assert sp.simplify(p2.expr - expected) == 0

    # m = 2 variables, partition [1, 1], alpha = 1
    # Schur polynomial s_[1,1] * 2 = 2*x1*x2
    p3 = jack_polynomial(2, [1, 1], 1)
    vy1, vy2 = p3.variables
    assert sp.simplify(p3.expr - 2 * vy1 * vy2) == 0
