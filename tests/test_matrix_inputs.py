from fractions import Fraction
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree.hyperbolic import MultiplicativeMatrixPencil, SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial


@pytest.mark.parametrize(
    "pencil_type", [SymmetricMatrixPencil, MultiplicativeMatrixPencil]
)
@pytest.mark.parametrize("third", [Fraction(1, 3), sp.Rational(1, 3), flint.fmpq(1, 3)])
def test_original_integer_and_rational_inputs(pencil_type: Any, third: Any) -> None:
    large = 2**53 + 1
    source = np.array([[large, 0], [0, third]], dtype=object)
    pencil = pencil_type([source])
    source[0, 0] = 0  # Construction copies the input.
    assert pencil.characteristic_polynomial([1]) == flint.fmpq_poly(
        [flint.fmpq(large, 3), -flint.fmpq(3 * large + 1, 3), 1]
    )
    assert pencil.evaluate([1]).dtype == np.float64
    assert pencil.evaluate([1])[0, 0] == float(large)


@pytest.mark.parametrize(
    "pencil_type", [SymmetricMatrixPencil, MultiplicativeMatrixPencil]
)
def test_float_input_preserves_its_binary_value(pencil_type: Any) -> None:
    value = 0.1
    numerator, denominator = value.as_integer_ratio()
    pencil = pencil_type([[[value]]])
    assert pencil.characteristic_polynomial([1])[0] == -flint.fmpq(
        numerator, denominator
    )
    # Precision already lost by the caller's float conversion cannot be recovered.
    rounded = pencil_type([np.array([[2**53 + 1]], dtype=np.float64)])
    assert rounded.characteristic_polynomial([1])[0] == -(2**53)


@pytest.mark.parametrize(
    "pencil_type", [SymmetricMatrixPencil, MultiplicativeMatrixPencil]
)
def test_exact_slp_uses_original_inputs(pencil_type: Any) -> None:
    large = 2**53 + 1
    pencil = pencil_type([[[large, 0], [0, Fraction(1, 3)]], np.eye(2)])
    slp = pencil.characteristic_polynomial_slp()
    # det(x*A + y*I) = (large*x+y)*(x/3+y).
    point = np.array([1, 2], dtype=np.float64)
    assert slp.evaluate(point, exact=True) == flint.fmpq(7 * (large + 2), 3)
    gradient = slp.gradient(point, exact=True)
    assert list(gradient) == [
        flint.fmpq(8 * large + 2, 3),
        flint.fmpq(3 * large + 13, 3),
    ]
    hessian = slp.hessian(point, exact=True)
    assert list(hessian.flat) == [
        flint.fmpq(2 * large, 3),
        flint.fmpq(3 * large + 1, 3),
        flint.fmpq(3 * large + 1, 3),
        2,
    ]


@pytest.mark.parametrize("method", ["direct", "interpolated", "sparse"])
@pytest.mark.parametrize("single_matrix", [False, True])
@pytest.mark.parametrize("large", [2**53 + 1, 2**80 + 1])
def test_multivariate_paths_preserve_original_entries(
    method: str, single_matrix: bool, large: int
) -> None:
    matrices = [[[large, 0], [0, Fraction(1, 3)]]]
    if not single_matrix:
        matrices.append([[1, 0], [0, 1]])
    pencil = SymmetricMatrixPencil(matrices)
    factory = {
        "direct": MultivariatePolynomial.from_symmetric_matrix_pencil,
        "interpolated": MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated,
        "sparse": MultivariatePolynomial.from_symmetric_matrix_pencil_sparse,
    }[method]
    result = factory(pencil)
    x = result.variables[0]
    expected = sp.Rational(large, 3) * x**2
    if not single_matrix:
        y = result.variables[1]
        expected += (large + sp.Rational(1, 3)) * x * y + y**2
    assert sp.expand(result.expr - expected) == 0


@pytest.mark.parametrize("large", [2**53 + 1, 2**80 + 1])
def test_diagonal_specialization_uses_original_entries(large: int) -> None:
    pencil = SymmetricMatrixPencil([np.eye(2), [[large, 0], [0, Fraction(1, 3)]]])
    result = pencil.diagonal_specialization([1, 0], [0, -1])
    assert list(result.coeffs) == [1, -large - sp.Rational(1, 3), sp.Rational(large, 3)]


@pytest.mark.parametrize(
    "pencil_type", [SymmetricMatrixPencil, MultiplicativeMatrixPencil]
)
@pytest.mark.parametrize(
    "matrices",
    [[], [[[1, 2]]], [np.eye(2), np.eye(3)], [[[np.nan]]], [[[np.inf]]], [[[1j]]]],
)
def test_invalid_matrix_inputs(pencil_type: Any, matrices: Any) -> None:
    with pytest.raises(ValueError):
        pencil_type(matrices)


def test_symmetry_is_checked_before_float_rounding() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        SymmetricMatrixPencil([[[1, 2**53 + 1], [2**53, 1]]])


@pytest.mark.parametrize(
    "pencil_type", [SymmetricMatrixPencil, MultiplicativeMatrixPencil]
)
def test_coordinate_lengths_match_in_both_paths(pencil_type: Any) -> None:
    pencil = pencil_type([np.eye(2), np.eye(2)])
    with pytest.raises(ValueError, match="Expected 2"):
        pencil.evaluate([1])
    with pytest.raises(ValueError, match="Expected 2"):
        pencil.characteristic_polynomial([1])
