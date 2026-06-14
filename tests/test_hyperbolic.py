import numpy as np
import sympy as sp

from finitefree.hyperbolic import MultiplicativeMatrixPencil, SymmetricMatrixPencil
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
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

    x, y = sp.symbols("x y")
    # P(x, y) = x^2 + 2*x*y + y^2 (homogeneous degree 2)
    p = MultivariatePolynomial(x**2 + 2 * x * y + y**2, [x, y])
    assert p.degree() == 2
    assert p.is_homogeneous() is True

    # Non-homogeneous check: x^2 + y
    p_non = MultivariatePolynomial(x**2 + y, [x, y])
    assert p_non.is_homogeneous() is False


def test_multivariate_directional_and_mixed_derivatives() -> None:
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

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
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

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
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

    # A1 = I_2, A2 = [[0, 1], [1, 0]]
    A1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    A2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    pencil = SymmetricMatrixPencil([A1, A2])

    p = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    x1, x2 = p.variables
    # determinant of [[x1, x2], [x2, x1]] = x1^2 - x2^2
    assert sp.simplify(p.expr - (x1**2 - x2**2)) == 0


def test_multivariate_from_pencil_sparse() -> None:
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

    A1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    A2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    pencil = SymmetricMatrixPencil([A1, A2])

    p_sparse = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(pencil)
    x1, x2 = p_sparse.variables
    assert sp.simplify(p_sparse.expr - (x1**2 - x2**2)) == 0


def test_diagonal_specialization() -> None:
    A1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    A2 = np.array([[0.0, 2.0], [2.0, 0.0]])
    pencil = SymmetricMatrixPencil([A1, A2])

    # A = w1 A1 + w2 A2 = [[w1, 2*w2], [2*w2, w1]]
    # B = b1 A1 + b2 A2 = [[b1, 2*b2], [2*b2, b1]]
    # det(z A + B) = (z*w1 + b1)^2 - 4*(z*w2 + b2)^2
    # Set w = (1, 1), b = (2, 3)
    # det(z A + B) = (z + 2)^2 - 4*(z + 3)^2
    # = z^2 + 4z + 4 - 4*(z^2 + 6z + 9) = -3z^2 - 20z - 32
    poly = pencil.diagonal_specialization([1, 1], [2, 3])
    # coefficients in descending order, monic normalized: [1, 20/3, 32/3]
    expected_coeffs = [1, sp.Rational(20, 3), sp.Rational(32, 3)]
    assert len(poly.coeffs) == 3
    for c, exp_c in zip(poly.coeffs, expected_coeffs):
        assert c == exp_c


def test_multiplicative_matrix_pencil() -> None:
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

    # A1 = [[1, 2], [3, 4]] (asymmetric)
    # A2 = [[0, 1], [0, 0]] (asymmetric)
    A1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    A2 = np.array([[0.0, 1.0], [0.0, 0.0]])
    pencil = MultiplicativeMatrixPencil([A1, A2])

    p = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    x1, x2 = p.variables
    # det of [[x1, 2*x1 + x2], [3*x1, 4*x1]]
    # = 4*x1^2 - 3*x1*(2*x1 + x2) = -2*x1^2 - 3*x1*x2
    assert sp.simplify(p.expr - (-2 * x1**2 - 3 * x1 * x2)) == 0


def test_one_dimensional_pencil() -> None:
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

    # m = 1 boundary case test
    A1 = np.array([[2.0, 3.0], [3.0, 5.0]])
    pencil = SymmetricMatrixPencil([A1])

    p_sparse = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(pencil)
    assert len(p_sparse.variables) == 1
    # det(x1 * A1) = x1^2 * (2 * 5 - 3 * 3) = 1 * x1^2
    assert sp.simplify(p_sparse.expr - p_sparse.variables[0] ** 2) == 0

    p_dense = MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(pencil)
    assert len(p_dense.variables) == 1
    assert sp.simplify(p_dense.expr - p_dense.variables[0] ** 2) == 0


def test_exact_hyperbolic_solvers() -> None:
    # A1 = I_2, A2 = [[0, 1], [1, 0]]
    A1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    A2 = np.array([[0.0, 1.0], [1.0, 0.0]])

    pencil = SymmetricMatrixPencil([A1, A2])

    # Exact verify_hyperbolicity
    assert pencil.verify_hyperbolicity([1.0, 0.0], exact=True) is True
    assert pencil.verify_hyperbolicity([0.0, 1.0], exact=True) is False

    slp = pencil.characteristic_polynomial_slp()
    x = [2.0, 1.0]

    # Exact evaluate
    val = slp.evaluate(x, exact=True)
    assert val == 3

    # Exact gradient
    grad = slp.gradient(x, exact=True)
    assert len(grad) == 2
    assert grad[0] == 4
    assert grad[1] == -2

    # Exact Hessian
    hess = slp.hessian(x, exact=True)
    assert hess.shape == (2, 2)
    assert hess[0, 0] == 2
    assert hess[1, 1] == -2
    assert hess[0, 1] == 0
    assert hess[1, 0] == 0


def test_interpolated_matrix_pencil() -> None:
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

    # Construct a 3x3 matrix pencil (size < 4)
    m, n = 3, 3
    matrices = [np.random.randint(-3, 4, size=(n, n)).astype(float) for _ in range(m)]
    matrices = [A + A.T for A in matrices]

    pencil = SymmetricMatrixPencil(matrices)

    # Compute using both Berkowitz determinant and grid interpolation
    mp_sym = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    mp_interp = MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(
        pencil, parallel=True
    )

    # Verify exact equality of symbolic expressions
    assert sp.simplify(mp_sym.expr - mp_interp.expr) == 0


def test_interpolated_matrix_pencil_large() -> None:
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

    # Scale up matrix pencil to 5x5 to challenge the CRT solver
    m, n = 3, 5
    matrices = [np.random.randint(-2, 2, size=(n, n)).astype(float) for _ in range(m)]
    matrices = [A + A.T for A in matrices]

    pencil = SymmetricMatrixPencil(matrices)

    # from_symmetric_matrix_pencil automatically routes to interpolated for n >= 4
    mp_auto = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    mp_interp = MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(
        pencil, parallel=False
    )

    assert sp.simplify(mp_auto.expr - mp_interp.expr) == 0


def test_to_fmpq_mpoly() -> None:
    import flint

    if not hasattr(flint, "fmpq_mpoly_ctx"):
        import pytest

        pytest.skip(
            "flint.fmpq_mpoly_ctx is not available in the installed python-flint version"
        )

    # Construct a multivariate polynomial P(x, y) = 3 x^2 + 2 x y + 5 y^2
    x, y = sp.symbols("x y")
    mp = MultivariatePolynomial(3 * x**2 + 2 * x * y + 5 * y**2, [x, y])

    # Convert to C-level fmpq_mpoly sparse array
    flint_poly = mp.to_fmpq_mpoly()

    assert isinstance(flint_poly, flint.fmpq_mpoly)

    # Evaluate at x = 2, y = 3
    assert flint_poly(2, 3) == 69


def test_parallel_diagonal_specialization() -> None:
    # Define simple symmetric matrices for a pencil
    A1 = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float)
    A2 = np.array([[1.0, 0.5], [0.5, 4.0]], dtype=float)

    pencil = SymmetricMatrixPencil([A1, A2])

    # Compute specialization sequentially and in parallel
    w = [1, 2]
    b = [0, 1]

    p_seq = pencil.diagonal_specialization(w, b, parallel=False)
    p_par = pencil.diagonal_specialization(w, b, parallel=True)

    # Coeffs should match exactly
    np.testing.assert_allclose(
        np.array(p_seq.coeffs, dtype=float),
        np.array(p_par.coeffs, dtype=float),
        rtol=1e-12,
    )


def test_custom_arithmetic_slp() -> None:
    from finitefree.hyperbolic.slp import StraightLineProgram

    # Test SLP: P(x0, x1) = x0^2 - x1^2
    ops = [
        ("mul", 0, 0),  # v2 = v0 * v0
        ("mul", 1, 1),  # v3 = v1 * v1
        ("sub", 2, 3),  # v4 = v2 - v3
    ]
    slp = StraightLineProgram(ops)
    x = np.array([2.0, 1.0])

    # Float evaluation and gradient
    val_float = slp.evaluate(x)
    grad_float = slp.gradient(x)
    assert np.isclose(val_float, 3.0)
    assert np.allclose(grad_float, [4.0, -2.0])

    # Exact evaluation and gradient
    val_exact = slp.evaluate(x, exact=True)
    grad_exact = slp.gradient(x, exact=True)
    assert val_exact == 3
    assert grad_exact[0] == 4
    assert grad_exact[1] == -2


def test_exact_characteristic_polynomial_interpolation() -> None:
    # A1 = I_2, A2 = [[0, 1], [1, 0]]
    A1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    A2 = np.array([[0.0, 1.0], [1.0, 0.0]])

    pencil = SymmetricMatrixPencil([A1, A2])
    x = [2.0, 1.0]

    # Evaluated matrix A(x) = [[2, 1], [1, 2]]
    # charpoly of A(x) is det(t*I - A(x)) = (t-2)^2 - 1 = t^2 - 4*t + 3
    p = pencil.characteristic_polynomial(x)

    import flint

    assert isinstance(p, flint.fmpq_poly)
    assert p.degree() == 2
    # Coefficients in ascending order: [3, -4, 1]
    assert list(p) == [flint.fmpq(3), flint.fmpq(-4), flint.fmpq(1)]


def test_multiplicative_slp_exact() -> None:
    A1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    A2 = np.array([[0.0, 1.0], [0.0, 0.0]])
    pencil = MultiplicativeMatrixPencil([A1, A2])
    x = [2.0, 1.0]

    slp = pencil.characteristic_polynomial_slp()
    # det of [[2, 5], [6, 8]] = 16 - 30 = -14
    val = slp.evaluate(x, exact=True)
    assert val == -14

    grad = slp.gradient(x, exact=True)
    # Gradient checks: det([[x1, 2*x1+x2], [3*x1, 4*x1]]) = -2*x1^2 - 3*x1*x2
    # grad_x1 = -4*x1 - 3*x2 = -8 - 3 = -11
    # grad_x2 = -3*x1 = -6
    assert len(grad) == 2
    assert grad[0] == -11
    assert grad[1] == -6
