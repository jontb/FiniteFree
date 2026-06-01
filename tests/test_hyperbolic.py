import numpy as np
import sympy as sp

from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial


def test_straight_line_program_and_pencil() -> None:
    # A1 = I_2, A2 = [[0, 1], [1, 0]]
    A1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    A2 = np.array([[0.0, 1.0], [1.0, 0.0]])

    pencil = SymmetricMatrixPencil([A1, A2])
    assert pencil.n == 2
    assert pencil.m == 2

    # Verify joint hyperbolicity: e = (1, 0) -> A(e) = I_2 > 0
    assert pencil.verify_hyperbolicity([1.0, 0.0]) is True
    # e = (0, 1) -> eigenvalues are -1 and 1
    assert pencil.verify_hyperbolicity([0.0, 1.0]) is False

    # SLP checks
    slp = pencil.characteristic_polynomial_slp()
    x = np.array([2.0, 1.0])  # A(x) = [[2, 1], [1, 2]] -> det = 3
    assert np.allclose(slp.evaluate(x), 3.0)

    # Gradient: d/dx1(x1^2 - x2^2) = 2*x1 = 4, d/dx2(x1^2 - x2^2) = -2*x2 = -2
    grad = slp.gradient(x)
    assert np.allclose(grad, [4.0, -2.0])

    # Hessian: d^2/dx1^2 = 2, d^2/dx2^2 = -2, mixed = 0
    hess = slp.hessian(x)
    expected_hess = np.array([[2.0, 0.0], [0.0, -2.0]])
    assert np.allclose(hess, expected_hess)


def test_multivariate_polynomial_homogeneous() -> None:
    x, y = sp.symbols("x y")
    # P(x, y) = x^2 + 2*x*y + y^2 (homogeneous degree 2)
    p = MultivariatePolynomial(x**2 + 2 * x * y + y**2, [x, y])
    assert p.degree() == 2
    assert p.is_homogeneous() is True

    # Non-homogeneous check: x^2 + y
    p_non = MultivariatePolynomial(x**2 + y, [x, y])
    assert p_non.is_homogeneous() is False


def test_multivariate_directional_and_mixed_derivatives() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x**2 + 2 * x * y + y**2, [x, y])

    # Directional derivative in e = (1, 1)
    # D_e P = dP/dx + dP/dy = (2x + 2y) + (2x + 2y) = 4x + 4y
    dp = p.directional_derivative([1, 1])
    assert dp.expr == 4 * x + 4 * y

    # Mixed partial derivative: d^2/dxdy P = 2
    d2p = p.mixed_partial_derivative([1, 1])
    assert d2p.expr == 2


def test_multivariate_normalized_coefficients() -> None:
    x, y = sp.symbols("x y")
    p = MultivariatePolynomial(x**2 + 2 * x * y + y**2, [x, y])

    # Normalized coefficients:
    # x^2 -> c_(2,0) = 1, multinomial(2, [2,0]) = 1 -> norm = 1
    # x*y -> c_(1,1) = 2, multinomial(2, [1,1]) = 2 -> norm = 1
    # y^2 -> c_(0,2) = 1, multinomial(2, [0,2]) = 1 -> norm = 1
    norm_coeffs = p.normalized_coefficients()
    assert norm_coeffs[(2, 0)] == 1
    assert norm_coeffs[(1, 1)] == 1
    assert norm_coeffs[(0, 2)] == 1


def test_multivariate_from_pencil() -> None:
    # A1 = I_2, A2 = [[0, 1], [1, 0]]
    A1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    A2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    pencil = SymmetricMatrixPencil([A1, A2])

    p = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    x1, x2 = p.variables
    # determinant of [[x1, x2], [x2, x1]] = x1^2 - x2^2
    assert sp.simplify(p.expr - (x1**2 - x2**2)) == 0
