import random
from typing import Any

import numpy as np
import pytest
import sympy as sp

import finitefree.multivariate as multivariate
from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial
from finitefree.utils.modular import prime_generator


def test_specialization_cancellation_cannot_hide_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancelSpecialization(random.Random):
        def __init__(self, seed: Any = None) -> None:
            super().__init__(seed)
            self.first = True

        def randint(self, a: int, b: int) -> int:
            if self.first:
                self.first = False
                return 2
            return super().randint(a, b)

    monkeypatch.setattr(random, "Random", CancelSpecialization)
    # At x2=2 and x3=1, x1*(x2-2*x3) vanishes identically in x1.
    # Every modular specialization then misses both true monomials.
    pencil = SymmetricMatrixPencil(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, -2]]]
    )
    result = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(pencil)
    x, y, z = result.variables
    assert result.expr == x * y - 2 * x * z


def test_failed_discovery_has_bounded_deterministic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SingularProbes:
        def __init__(self, seed: Any = None) -> None:
            pass

        def randint(self, a: int, b: int) -> int:
            return 2

    monkeypatch.setattr(random, "Random", SingularProbes)

    def bounded_primes(start: int) -> Any:
        for i, prime in enumerate(prime_generator(start)):
            if i >= 8:
                pytest.fail("Sparse discovery retried more than eight failed primes")
            yield prime

    monkeypatch.setattr(multivariate, "prime_generator", bounded_primes)
    pencil = SymmetricMatrixPencil([[[1, 0], [0, 2]], [[3, 0], [0, 4]]])
    result = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(pencil)
    x, y = result.variables
    assert result.expr == sp.expand((x + 3 * y) * (2 * x + 4 * y))


@pytest.mark.parametrize("mutation", ["missing", "wrong", "empty", "collision"])
def test_certificate_repairs_incomplete_or_incorrect_candidates(
    mutation: str,
) -> None:
    matrices = [
        [[2, 1, 0], [1, -3, 2], [0, 2, 1]],
        [[-1, 0, 3], [0, 2, 0], [3, 0, 4]],
        [[3, -2, 1], [-2, 1, 0], [1, 0, -1]],
    ]
    x, y, z = sp.symbols("x y z")
    expected = sp.Poly(
        (
            x * sp.Matrix(matrices[0])
            + y * sp.Matrix(matrices[1])
            + z * sp.Matrix(matrices[2])
        ).det(),
        x,
        y,
        z,
    )
    candidate = {exps[:-1]: int(c) for exps, c in expected.terms()}
    first = next(iter(candidate))
    if mutation == "missing":
        del candidate[first]
    elif mutation == "wrong":
        candidate[first] += 1
    elif mutation == "empty":
        candidate.clear()
    bound = multivariate._determinant_coefficient_bound(matrices)
    base, weights = multivariate._determinant_encoding_parameters(3, 3, bound, 1000000)
    if mutation == "collision":
        # Equal encoded values alone are insufficient without coefficient bounds:
        # shifting b units from the constant digit to the next digit preserves it.
        candidate[(0, 0)] = candidate.get((0, 0), 0) - base
        candidate[(1, 0)] = candidate.get((1, 0), 0) + 1
    repaired = multivariate._certify_sparse_determinant_coefficients(
        matrices, candidate, bound, base, weights
    )
    actual = {exps + (3 - sum(exps),): c for exps, c in repaired.items()}
    assert actual == {exps: int(c) for exps, c in expected.terms()}


def test_complete_sparse_candidate_avoids_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_decode(*args: Any) -> Any:
        pytest.fail("An independently verified sparse candidate was decoded again")

    monkeypatch.setattr(multivariate, "_decode_determinant_encoding", unexpected_decode)
    pencil = SymmetricMatrixPencil(
        [
            np.diag([1, 1, 0, 0]),
            np.diag([0, 0, 1, 1]),
            np.zeros((4, 4), dtype=int),
            np.zeros((4, 4), dtype=int),
        ]
    )
    result = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(pencil)
    x, y, _, _ = result.variables
    assert result.expr == x**2 * y**2


def test_verification_limit_is_checked_before_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_random(*args: Any) -> Any:
        pytest.fail("Randomized discovery started before the verification size check")

    monkeypatch.setattr(random, "Random", unexpected_random)
    pencil = SymmetricMatrixPencil([np.eye(3, dtype=int)] * 4)
    with pytest.raises(ValueError, match="max_verification_bits"):
        MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(
            pencil, max_verification_bits=10
        )


@pytest.mark.parametrize("bad", [0, -1, True, np.bool_(True), 1.5, "1000"])
def test_invalid_verification_limit(bad: Any) -> None:
    pencil = SymmetricMatrixPencil([[[1]], [[2]]])
    with pytest.raises((TypeError, ValueError)):
        MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(
            pencil, max_verification_bits=bad
        )


def test_explicit_larger_limit_preserves_rational_coefficients() -> None:
    matrices = [
        sp.diag(sp.Rational(1, 3), sp.Rational(-2, 7)),
        sp.Matrix([[sp.Rational(2, 5), 1], [1, sp.Rational(3, 11)]]),
        sp.diag(sp.Rational(-5, 13), sp.Rational(7, 17)),
    ]
    pencil = SymmetricMatrixPencil([a.tolist() for a in matrices])
    with pytest.raises(ValueError, match="max_verification_bits"):
        MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(
            pencil, max_verification_bits=1
        )
    result = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(
        pencil, max_verification_bits=np.int64(1000000)
    )
    matrix = sum((v * a for v, a in zip(result.variables, matrices)), sp.zeros(2))
    assert result.expr == sp.expand(matrix.det())


def test_zero_row_bound_needs_no_encoding_or_random_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_random(*args: Any) -> Any:
        pytest.fail("A proved zero determinant started randomized discovery")

    monkeypatch.setattr(random, "Random", unexpected_random)
    pencil = SymmetricMatrixPencil([np.diag([1, 0])] * 8)
    result = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(
        pencil, max_verification_bits=1
    )
    assert result.expr == 0


def test_empty_pencil_determinant_is_one() -> None:
    pencil = SymmetricMatrixPencil([np.empty((0, 0))] * 3)
    result = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(
        pencil, max_verification_bits=1
    )
    assert result.expr == 1


def test_negative_determinant_and_zero_encoding_digits() -> None:
    matrices = [
        sp.diag(-1, 2, 0),
        sp.diag(0, 0, 3),
        sp.zeros(3),
        sp.zeros(3),
    ]
    integer_matrices = [[[int(c) for c in row] for row in a.tolist()] for a in matrices]
    bound = multivariate._determinant_coefficient_bound(integer_matrices)
    base, weights = multivariate._determinant_encoding_parameters(3, 4, bound, 1000000)
    result = multivariate._certify_sparse_determinant_coefficients(
        integer_matrices, {}, bound, base, weights
    )
    assert result == {(2, 1, 0): -6}


def test_many_variables_fail_before_constructing_huge_powers() -> None:
    pencil = SymmetricMatrixPencil([[[1]]] * 100)
    with pytest.raises(ValueError, match="max_verification_bits"):
        MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(pencil)
