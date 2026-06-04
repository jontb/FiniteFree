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


def test_jacobi_polynomials() -> None:
    # Degree 0
    p0 = jacobi_polynomial(0, 1, 1)
    assert p0.degree == 0
    assert p0.coeffs[0] == 1

    # Degree 1
    # P_1^(a, b)(x) = (a - b)/2 + (a + b + 2)/2 * x
    # For a = 1, b = 2: (1 - 2)/2 + (1 + 2 + 2)/2 * x = -1/2 + 5/2 * x
    # Monic version: x - 1/5. Descending coefficients: [1, -1/5]
    p1 = jacobi_polynomial(1, 1, 2)
    assert p1.degree == 1
    assert p1.coeffs[0] == 1
    assert p1.coeffs[1] == sp.Rational(-1, 5)

    # Degree 2
    # For a = 0, b = 0 (Legendre Polynomials P_2(x) = 1/2 * (3x^2 - 1))
    # Monic: x^2 - 1/3. Descending coefficients: [1, 0, -1/3]
    p2 = jacobi_polynomial(2, 0, 0)
    assert p2.degree == 2
    assert p2.coeffs[0] == 1
    assert p2.coeffs[1] == 0
    assert p2.coeffs[2] == sp.Rational(-1, 3)


def test_hahn_polynomials() -> None:
    # Degree 0
    h0 = hahn_polynomial(0, 1, 1, 5)
    assert h0.degree == 0
    assert h0.coeffs[0] == 1

    # Degree 1
    # Q_1(x; a, b, N) = 1 - (a + b + 2)/((a + 1)*N) * x
    # For a=1, b=1, N=4: 1 - (1 + 1 + 2)/((1 + 1)*4) * x = 1 - 4/8 * x = 1 - 1/2 * x
    # Monic: x - 2. Descending coeffs: [1, -2]
    h1 = hahn_polynomial(1, 1, 1, 4)
    assert h1.degree == 1
    assert h1.coeffs[0] == 1
    assert h1.coeffs[1] == -2


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


def test_hermite_polynomials() -> None:
    # 1. Physicist Hermite H_2(x) = 4x^2 - 2
    # Monic version: x^2 - 1/2. Descending coeffs: [1, 0, -1/2]
    h2 = hermite_polynomial(2, physicist=True)
    assert h2.degree == 2
    assert h2.coeffs[0] == 1
    assert h2.coeffs[1] == 0
    assert h2.coeffs[2] == sp.Rational(-1, 2)

    # 2. Probabilist Hermite He_3(x) = x^3 - 3x
    # Monic version is the same. Descending coeffs: [1, 0, -3, 0]
    he3 = hermite_polynomial(3, physicist=False)
    assert he3.degree == 3
    assert he3.coeffs[0] == 1
    assert he3.coeffs[1] == 0
    assert he3.coeffs[2] == -3
    assert he3.coeffs[3] == 0


def test_laguerre_polynomials() -> None:
    # Generalized Laguerre L_2^(1)(x) = x^2/2 - 3x + 3
    # Monic version: x^2 - 6x + 6. Descending coeffs: [1, -6, 6]
    lag2 = laguerre_polynomial(2, 1)
    assert lag2.degree == 2
    assert lag2.coeffs[0] == 1
    assert lag2.coeffs[1] == -6
    assert lag2.coeffs[2] == 6

    # Test error cases
    with pytest.raises(ValueError):
        laguerre_polynomial(-1, 1)


def test_krawtchouk_polynomials() -> None:
    # Krawtchouk polynomial K_2(x; 1/2, 4)
    # Hypergeometric definition:
    # 2F1(-2, -x; -4; 2) = 1 + (-2)*(-x)/(-4)*2 + (-2)*(-1)*(-x)*(-x+1)/((-4)*(-3))*4
    # = 1 - x + x(x-1)/3 = 1 - x + x^2/3 - x/3 = x^2/3 - 4x/3 + 1
    # Monic version: x^2 - 4x + 3. Descending coeffs: [1, -4, 3]
    import pytest

    k2 = krawtchouk_polynomial(2, sp.Rational(1, 2), 4)
    assert k2.degree == 2
    assert k2.coeffs[0] == 1
    assert k2.coeffs[1] == -4
    assert k2.coeffs[2] == 3

    # Test error cases
    with pytest.raises(ValueError):
        krawtchouk_polynomial(-1, 0.5, 4)
    with pytest.raises(ValueError):
        krawtchouk_polynomial(5, 0.5, 4)
    with pytest.raises(ValueError):
        krawtchouk_polynomial(2, 0.0, 4)


def test_chebyshev_polynomials() -> None:
    # 1. T_2(x) = 2x^2 - 1. Monic: x^2 - 1/2. Descending: [1, 0, -1/2]
    t2 = chebyshev_t_polynomial(2)
    assert t2.degree == 2
    assert t2.coeffs[0] == 1
    assert t2.coeffs[1] == 0
    assert t2.coeffs[2] == sp.Rational(-1, 2)

    # 2. U_2(x) = 4x^2 - 1. Monic: x^2 - 1/4. Descending: [1, 0, -1/4]
    u2 = chebyshev_u_polynomial(2)
    assert u2.degree == 2
    assert u2.coeffs[0] == 1
    assert u2.coeffs[1] == 0
    assert u2.coeffs[2] == sp.Rational(-1, 4)

    with pytest.raises(ValueError):
        chebyshev_t_polynomial(-1)
    with pytest.raises(ValueError):
        chebyshev_u_polynomial(-1)


def test_legendre_polynomials() -> None:
    # P_3(x) = 1/2 * (5x^3 - 3x). Monic: x^3 - 3/5 x. Descending: [1, 0, -3/5, 0]
    leg3 = legendre_polynomial(3)
    assert leg3.degree == 3
    assert leg3.coeffs[0] == 1
    assert leg3.coeffs[1] == 0
    assert leg3.coeffs[2] == sp.Rational(-3, 5)
    assert leg3.coeffs[3] == 0

    with pytest.raises(ValueError):
        legendre_polynomial(-1)
