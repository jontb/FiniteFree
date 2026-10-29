from typing import Any, Callable, Sequence

import flint
import numpy as np
import sympy as sp

from finitefree.hyperbolic import MultiplicativeMatrixPencil, SymmetricMatrixPencil
from finitefree.hyperbolic.slp import StraightLineProgram
from finitefree.multivariate import MultivariatePolynomial
from tests._cases import case

IDENTITY = np.array([[1.0, 0.0], [0.0, 1.0]])
SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])
ASYMMETRIC = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.0, 1.0], [0.0, 0.0]])]


def _symbolic_determinant(
    matrices: Sequence[np.ndarray], variables: Sequence[sp.Symbol]
) -> sp.Expr:
    """Independent SymPy Berkowitz determinant of sum_i x_i A_i."""
    size = matrices[0].shape[0]
    total = sp.zeros(size)
    for variable, matrix in zip(variables, matrices):
        entries = [sp.Rational(value) for value in matrix.ravel()]
        total += variable * sp.Matrix(size, size, entries)
    return total.det(method="berkowitz")


def test_pencil_shape_and_hyperbolicity_direction() -> None:
    pencil = SymmetricMatrixPencil([IDENTITY, SWAP])
    assert pencil.n == 2
    assert pencil.m == 2
    for exact in [False, True]:
        with case(exact=exact):
            # A(1, 0) = I is positive definite; A(0, 1) has eigenvalues -1 and 1.
            assert pencil.verify_hyperbolicity([1.0, 0.0], exact=exact) is True
            assert pencil.verify_hyperbolicity([0.0, 1.0], exact=exact) is False


def test_slp_values_and_derivatives_against_closed_forms() -> None:
    # det(x1 I + x2 SWAP) = x1^2 - x2^2; the explicit program computes the same
    # value and gradient. Hessians need a pencil, so that program has none.
    # det(x1 A1 + x2 A2) = -2 x1^2 - 3 x1 x2 for the asymmetric pair.
    Expected = tuple[int, list[int], list[list[int]] | None]
    programs: dict[str, tuple[Any, Expected]] = {
        "symmetric pencil": (
            SymmetricMatrixPencil([IDENTITY, SWAP]).characteristic_polynomial_slp(),
            (3, [4, -2], [[2, 0], [0, -2]]),
        ),
        "explicit program": (
            StraightLineProgram([("mul", 0, 0), ("mul", 1, 1), ("sub", 2, 3)]),
            (3, [4, -2], None),
        ),
        "multiplicative pencil": (
            MultiplicativeMatrixPencil(ASYMMETRIC).characteristic_polynomial_slp(),
            (-14, [-11, -6], [[-4, -3], [-3, 0]]),
        ),
    }
    for name, (slp, (value, gradient, hessian)) in programs.items():
        with case(program=name, exact=False):
            x = np.array([2.0, 1.0])
            assert np.isclose(slp.evaluate(x), value)
            np.testing.assert_allclose(slp.gradient(x), gradient)
            if hessian is not None:
                np.testing.assert_allclose(slp.hessian(x), hessian, atol=1e-12)
        with case(program=name, exact=True):
            point = [2.0, 1.0]
            assert slp.evaluate(point, exact=True) == value
            exact_gradient = slp.gradient(point, exact=True)
            assert len(exact_gradient) == 2
            assert list(exact_gradient) == gradient
            if hessian is not None:
                exact_hessian = slp.hessian(point, exact=True)
                assert exact_hessian.shape == (2, 2)
                assert exact_hessian.tolist() == hessian


def test_multivariate_structure_of_fixed_quadratics() -> None:
    x, y = sp.symbols("x y")
    square = MultivariatePolynomial(x**2 + 2 * x * y + y**2, [x, y])
    assert square.degree() == 2
    assert square.is_homogeneous() is True
    assert MultivariatePolynomial(x**2 + y, [x, y]).is_homogeneous() is False

    # D_(1,1) P = 4x + 4y and d^2 P / dx dy = 2.
    assert square.directional_derivative([1, 1]).expr == 4 * x + 4 * y
    assert square.mixed_partial_derivative([1, 1]).expr == 2

    # Each coefficient divided by its multinomial weight: 1/1, 2/2 and 1/1.
    normalized = square.normalized_coefficients()
    assert normalized[(2, 0)] == 1
    assert normalized[(1, 1)] == 1
    assert normalized[(0, 2)] == 1

    flint_poly = MultivariatePolynomial(
        3 * x**2 + 2 * x * y + 5 * y**2, [x, y]
    ).to_fmpq_mpoly()
    assert isinstance(flint_poly, flint.fmpq_mpoly)
    assert flint_poly(2, 3) == 69


def _seeded_symmetric_integer_pencil(size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    matrices = [rng.integers(-3, 4, size=(size, size)).astype(float) for _ in range(3)]
    return [matrix + matrix.T for matrix in matrices]


def test_pencil_determinant_constructions_against_independent_determinants() -> None:
    def fixed(form: Callable[..., sp.Expr]) -> Callable[..., sp.Expr]:
        return lambda matrices, variables: form(*variables)

    pencils: dict[str, tuple[Any, Callable[..., sp.Expr]]] = {
        "identity and swap": (
            SymmetricMatrixPencil([IDENTITY, SWAP]),
            fixed(lambda x1, x2: x1**2 - x2**2),
        ),
        "asymmetric multiplicative": (
            MultiplicativeMatrixPencil(ASYMMETRIC),
            fixed(lambda x1, x2: -2 * x1**2 - 3 * x1 * x2),
        ),
        # det(x1 A1) = x1^2 det(A1) = x1^2 for the one-variable boundary.
        "one variable": (
            SymmetricMatrixPencil([np.array([[2.0, 3.0], [3.0, 5.0]])]),
            fixed(lambda x1: x1**2),
        ),
        # Size five exceeds the Berkowitz threshold and routes to interpolation.
        "seeded 3x3": (
            SymmetricMatrixPencil(_seeded_symmetric_integer_pencil(3, 865)),
            _symbolic_determinant,
        ),
        "seeded 5x5": (
            SymmetricMatrixPencil(_seeded_symmetric_integer_pencil(5, 867)),
            _symbolic_determinant,
        ),
    }
    constructions: dict[str, Callable[[Any], MultivariatePolynomial]] = {
        "default": MultivariatePolynomial.from_symmetric_matrix_pencil,
        "sparse": MultivariatePolynomial.from_symmetric_matrix_pencil_sparse,
        "interpolated": lambda pencil: (
            MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(
                pencil, parallel=False
            )
        ),
        "parallel interpolated": lambda pencil: (
            MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(
                pencil, parallel=True
            )
        ),
    }
    for name, (pencil, reference) in pencils.items():
        for method, construct in constructions.items():
            with case(pencil=name, construction=method):
                p = construct(pencil)
                assert len(p.variables) == pencil.m
                expected = reference(pencil.matrices, p.variables)
                assert sp.expand(p.expr - expected) == 0


def test_diagonal_specialization_paths_against_closed_forms() -> None:
    # det(z(A1 + A2) + 2A1 + 3A2) = -3z^2 - 20z - 32 for A2 = 2 SWAP.
    # det(z(A1 + 2A2) + A2) = 43z^2 + 26z + 15/4 for the second pencil.
    specializations = {
        "scaled swap": (
            SymmetricMatrixPencil([IDENTITY, 2 * SWAP]),
            [1, 1],
            [2, 3],
            [1, sp.Rational(20, 3), sp.Rational(32, 3)],
        ),
        "dense pair": (
            SymmetricMatrixPencil(
                [
                    np.array([[2.0, 0.0], [0.0, 3.0]]),
                    np.array([[1.0, 0.5], [0.5, 4.0]]),
                ]
            ),
            [1, 2],
            [0, 1],
            [1, sp.Rational(26, 43), sp.Rational(15, 172)],
        ),
    }
    for name, (pencil, w, b, expected) in specializations.items():
        for parallel in [False, True]:
            with case(pencil=name, parallel=parallel):
                poly = pencil.diagonal_specialization(w, b, parallel=parallel)
                assert list(poly.coeffs) == expected


def test_exact_characteristic_polynomial_interpolation() -> None:
    pencil = SymmetricMatrixPencil([IDENTITY, SWAP])
    # A(2, 1) = [[2, 1], [1, 2]] has det(tI - A) = (t - 2)^2 - 1 = t^2 - 4t + 3.
    p = pencil.characteristic_polynomial([2.0, 1.0])
    assert isinstance(p, flint.fmpq_poly)
    assert p.degree() == 2
    assert list(p) == [flint.fmpq(3), flint.fmpq(-4), flint.fmpq(1)]
