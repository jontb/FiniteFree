import itertools
import math

import pytest
import sympy as sp

from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.multivariate import (
    MultivariatePolynomial,
    _determinant_coefficient_bound,
)
from finitefree.utils.modular import prime_generator


@pytest.mark.parametrize("method", ["interpolated", "sparse"])
def test_crt_agreement_cannot_hide_a_large_nonzero_coefficient(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force two four-prime batches for the grid path, and three primes for sparse.
    monkeypatch.setattr("finitefree.utils.parallel.os.cpu_count", lambda: 4)
    count = 8 if method == "interpolated" else 3
    coefficient = math.prod(itertools.islice(prime_generator(1000000007), count))
    pencil = SymmetricMatrixPencil([[[coefficient, 0], [0, 1]], [[1, 0], [0, 1]]])
    factory = (
        MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated
        if method == "interpolated"
        else MultivariatePolynomial.from_symmetric_matrix_pencil_sparse
    )
    result = factory(pencil)
    x, y = result.variables
    assert result.expr == sp.expand((coefficient * x + y) * (x + y))


@pytest.mark.parametrize("method", ["interpolated", "sparse"])
def test_large_rational_and_negative_coefficients_reconstruct(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("finitefree.utils.parallel.os.cpu_count", lambda: 2)
    coefficient = -(10**100 + 17)
    matrices = [
        sp.Matrix([[sp.Rational(coefficient, 13), 0], [0, sp.Rational(2, 7)]]),
        sp.Matrix([[sp.Rational(3, 5), 1], [1, sp.Rational(-1, 3)]]),
    ]
    pencil = SymmetricMatrixPencil([m.tolist() for m in matrices])
    factory = (
        MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated
        if method == "interpolated"
        else MultivariatePolynomial.from_symmetric_matrix_pencil_sparse
    )
    result = factory(pencil)
    x, y = result.variables
    assert result.expr == sp.expand((x * matrices[0] + y * matrices[1]).det())


@pytest.mark.parametrize("n,m", [(1, 2), (2, 2), (3, 2), (2, 3)])
def test_bound_covers_independently_expanded_coefficients(n: int, m: int) -> None:
    variables = sp.symbols(f"x0:{m}")
    matrices = [
        [[(r * 3 + c * 2 + j * 5) % 9 - 4 for c in range(n)] for r in range(n)]
        for j in range(m)
    ]
    symbolic = sum(
        (x * sp.Matrix(A) for x, A in zip(variables, matrices)), sp.zeros(n)
    ).det()
    coefficients = sp.Poly(symbolic, *variables).coeffs()
    bound = _determinant_coefficient_bound(matrices)
    assert all(abs(c) <= bound for c in coefficients)
    assert _determinant_coefficient_bound([[[0] * n for _ in range(n)]]) == 0
