"""Determinant SLP derivatives, SLP operation validation and empty pencils.

References are independent SymPy determinants and derivatives evaluated at the
exact binary values of the float64 coordinates. CRT loops are bounded by a
prime generator that fails the test (a BaseException, so no ``except
Exception`` can swallow it) instead of letting a regression hang.
"""

from typing import Any, Iterator

import numpy as np
import pytest
import sympy as sp
from numpy.typing import NDArray

import finitefree.hyperbolic.pencils as pencils
import finitefree.multivariate as multivariate
import finitefree.utils.parallel as parallel
from finitefree.hyperbolic import StraightLineProgram, SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial
from finitefree.utils.modular import _as_modular_array, prime_generator
from tests._cases import case

_MAX_TEST_PRIMES = 32


def _bounded_primes(start: int) -> Iterator[int]:
    for i, prime in enumerate(prime_generator(start)):
        if i >= _MAX_TEST_PRIMES:
            pytest.fail(f"Modular reconstruction requested more than {i} primes")
        yield prime


@pytest.fixture
def bounded_crt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("finitefree.utils.parallel.os.cpu_count", lambda: 4)
    monkeypatch.setattr(multivariate, "prime_generator", _bounded_primes)
    monkeypatch.setattr(pencils, "prime_generator", _bounded_primes)


def _diagonal_pencil(m: int) -> SymmetricMatrixPencil:
    return SymmetricMatrixPencil([np.diag(np.eye(m)[i]) for i in range(m)])


def _sympy_derivatives(
    matrices: list[Any], point: tuple[float, ...]
) -> tuple[list[sp.Rational], list[list[sp.Rational]]]:
    symbols = sp.symbols(f"x0:{len(matrices)}")
    determinant = sum(
        (s * sp.Matrix(A) for s, A in zip(symbols, matrices)),
        sp.zeros(len(matrices[0])),
    ).det()
    values = {
        s: sp.Rational(*float(v).as_integer_ratio()) for s, v in zip(symbols, point)
    }
    gradient = [sp.diff(determinant, s).subs(values) for s in symbols]
    hessian = [
        [sp.diff(determinant, s, t).subs(values) for t in symbols] for s in symbols
    ]
    return gradient, hessian


def _rational(value: Any) -> sp.Rational:
    # Convert FLINT rationals explicitly; implicit SymPy conversion can round.
    return sp.Rational(int(value.p), int(value.q))


def _as_float(values: Any) -> NDArray[np.float64]:
    return np.array(values, dtype=object).astype(np.float64)


def test_float_determinant_derivatives_at_tiny_nonsingular_determinants() -> None:
    # Diagonal pencils are perfectly conditioned entrywise, so float64 results
    # must agree with the exact derivatives at these points to rounding. The
    # old absolute |det| < 1e-15 pseudoinverse switch zeroed genuine terms, and
    # det underflow (1e-360) zeroed every derivative.
    point: tuple[float, ...]
    for point in [
        (1.0, 1e-16),
        (1.0, 2.0**-60),
        (3.0, 1e-15),
        (1e-200, 1e-200),
        (1.0, 1e-16, 1e-300),
        (1e-120, 1e-120, 1e-120),
        (-2.0, 1e-170, 1e-170),
    ]:
        with case(point=point):
            pencil = _diagonal_pencil(len(point))
            matrices = [
                np.diag(np.eye(len(point))[i]).tolist() for i in range(len(point))
            ]
            gradient, hessian = _sympy_derivatives(matrices, point)
            slp = pencil.characteristic_polynomial_slp()
            x = np.array(point)
            np.testing.assert_allclose(
                slp.gradient(x), _as_float(gradient), rtol=1e-12, atol=0
            )
            np.testing.assert_allclose(
                slp.hessian(x), _as_float(hessian), rtol=1e-12, atol=0
            )


def test_float_determinant_derivatives_reject_exactly_singular_points() -> None:
    # det = x0*x1 at (1, 0) has gradient (0, 1): pseudoinverse scaling by the
    # zero determinant returned the wrong finite value (0, 0). Like prepared
    # pencils and exact SLP, numerical SLP derivatives now reject the point.
    pencil = _diagonal_pencil(2)
    slp = pencil.characteristic_polynomial_slp()
    x = np.array([1.0, 0.0])
    method: str
    for method in ["gradient", "hessian"]:
        with case(method=method):
            # LinAlgError subclasses ValueError only from NumPy 2.0, so callers
            # on the supported NumPy 1.24 floor must catch LinAlgError itself.
            with pytest.raises(np.linalg.LinAlgError, match="nonsingular"):
                getattr(slp, method)(x)
    with pytest.raises(np.linalg.LinAlgError):
        pencil.prepare_numeric().factor(x).logabsdet_gradient()
    # The documented singular-point alternative differentiates the polynomial.
    polynomial = MultivariatePolynomial.from_symmetric_matrix_pencil(pencil)
    np.testing.assert_array_equal(polynomial.gradient_float64(x), [0.0, 1.0])
    assert slp.evaluate(x) == 0.0


def test_determinant_program_accepts_any_sequence_spelling() -> None:
    pencil = _diagonal_pencil(2)
    operations: Any
    for operations in [["det"], ("det",), np.array(["det"], dtype=object)]:
        with case(operations=operations):
            slp = StraightLineProgram(operations, pencil=pencil)
            # det(diag(2, 3)) = 6; d/dx (x0 x1) = (x1, x0); Hessian [[0,1],[1,0]].
            assert slp.evaluate(np.array([2.0, 3.0])) == 6.0
            np.testing.assert_array_equal(slp.gradient(np.array([2.0, 3.0])), [3, 2])
            np.testing.assert_array_equal(
                slp.hessian(np.array([2.0, 3.0])), [[0, 1], [1, 0]]
            )
            rational_point: Any = [2, 3]
            assert slp.evaluate(rational_point, exact=True) == 6
            assert list(slp.gradient(rational_point, exact=True)) == [3, 2]


def test_generic_programs_match_independent_symbolic_derivatives() -> None:
    x0, x1 = sp.symbols("x0 x1")
    expression = -((x0 + x1) * (x0 + x1) - 3) / x1
    operations = [
        ("+", 0, 1),
        ("mul", 2, 2),
        ("const", sp.Rational(3)),
        ("sub", 3, 4),
        ("/", 5, 1),
        ("neg", 6),
    ]
    point: Any = (2, 5)
    values = dict(zip((x0, x1), point))
    expected_value = expression.subs(values)
    expected_gradient = [sp.diff(expression, s).subs(values) for s in (x0, x1)]
    slp = StraightLineProgram(operations)
    operations.append(("pow", 0, 0))  # The program owns a validated snapshot.
    assert _rational(slp.evaluate(point, exact=True)) == expected_value
    assert [_rational(v) for v in slp.gradient(point, exact=True)] == expected_gradient
    np.testing.assert_allclose(
        slp.evaluate(np.array(point, dtype=float)), float(expected_value)
    )
    np.testing.assert_allclose(
        slp.gradient(np.array(point, dtype=float)), _as_float(expected_gradient)
    )


def test_invalid_operations_are_rejected_at_construction() -> None:
    operations: Any
    for operations, message in [
        ([("mul", 0, 1), ("pow", 2, 2)], "unknown code 'pow'"),
        ([("add", 0, 1), ("mult", 2, 2)], "unknown code 'mult'"),
        (["det", ("add", 0, 1)], "must be a tuple"),
        ([("det",)], "unknown code 'det'"),
        ([()], "empty"),
        ([(None, 0, 1)], "unknown code None"),
        ([("add", 0)], "exactly 2"),
        ([("neg", 0, 1)], "exactly 1"),
        ([("const",)], "exactly one value"),
        ([("add", 0, -1)], "nonnegative"),
        ([("add", 0, True)], "boolean"),
        ([("add", 0, 1.0)], "integers"),
        ("det", "sequence of operations"),
    ]:
        with case(operations=operations):
            with pytest.raises(ValueError, match=message):
                StraightLineProgram(operations)


def test_operand_indices_must_exist_for_the_supplied_inputs() -> None:
    slp = StraightLineProgram([("add", 0, 3), ("mul", 4, 1)])
    assert slp.evaluate(np.array([1.0, 2.0, 3.0, 4.0])) == 10.0
    short: Any = [1, 2, 3]
    exact: bool
    for exact in [False, True]:
        with case(exact=exact):
            with pytest.raises(ValueError, match="reference 4 input coordinates"):
                slp.evaluate(short, exact=exact)
            with pytest.raises(ValueError, match="reference 4 input coordinates"):
                slp.gradient(short, exact=exact)


def test_generic_hessian_is_explicitly_unsupported() -> None:
    pencil = _diagonal_pencil(2)
    # Previously a pencil made hessian() return the determinant Hessian for
    # p = x0 + x1, whose true Hessian is zero.
    for supplied in [None, pencil]:
        with case(pencil=supplied):
            slp = StraightLineProgram([("add", 0, 1)], pencil=supplied)
            assert slp.evaluate(np.array([2.0, 3.0])) == 5.0
            exact: bool
            for exact in [False, True]:
                with pytest.raises(NotImplementedError, match="generic"):
                    slp.hessian(np.array([2.0, 3.0]), exact=exact)


def test_modular_arrays_keep_empty_matrix_dimensions() -> None:
    p = 1000000007
    assert _as_modular_array([[], []], p, (2, 0, 0)).shape == (2, 0, 0)
    assert _as_modular_array([], p, (0, 0)).shape == (0, 0)
    reduced = _as_modular_array([[[-1, p + 2]]], p, (1, 1, 2))
    assert reduced.dtype == np.int64
    assert reduced.tolist() == [[[p - 1, 2]]]


def test_empty_matrix_workers_return_the_unit_determinant() -> None:
    p = 1000000007
    exponents = multivariate.get_monomial_exponents(2, 0)
    grid = multivariate.get_grid_points(1, (0,))
    assert parallel._eval_prime_worker(p, [[], []], grid, 0, 2, exponents) == (p, [1])
    assert parallel._eval_diagonal_specialization_prime_worker(p, [], [], 0, 0) == (
        p,
        [1],
    )


def test_empty_pencils_have_unit_determinant_in_every_constructor(
    bounded_crt: None,
) -> None:
    m: int
    for m in [1, 2, 3]:
        pencil = SymmetricMatrixPencil([np.zeros((0, 0))] * m)
        for factory in [
            MultivariatePolynomial.from_symmetric_matrix_pencil,
            MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated,
            MultivariatePolynomial.from_symmetric_matrix_pencil_sparse,
        ]:
            with case(m=m, factory=factory.__name__):
                result = factory(pencil)
                assert result.expr == 1
                assert len(result.variables) == m


def test_empty_pencil_specialization_is_the_unit_polynomial(
    bounded_crt: None,
) -> None:
    m: int
    for m in [1, 2, 3]:
        pencil = SymmetricMatrixPencil([np.zeros((0, 0))] * m)
        for parallel_flag in [False, True]:
            with case(m=m, parallel=parallel_flag):
                specialized = pencil.diagonal_specialization(
                    list(range(1, m + 1)), [0] * m, parallel=parallel_flag
                )
                assert list(specialized.coeffs) == [1]


class _WorkerDefect(Exception):
    pass


def _raise(error: BaseException) -> Any:
    def raiser(*args: Any, **kwargs: Any) -> Any:
        raise error

    return raiser


def _interpolated(pencil: SymmetricMatrixPencil) -> Any:
    return MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(pencil)


def _specialized(pencil: SymmetricMatrixPencil) -> Any:
    return pencil.diagonal_specialization([1, 2], [3, -1])


def test_persistent_prime_failures_surface_instead_of_retrying(
    bounded_crt: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pencil = SymmetricMatrixPencil([[[1, 0], [0, 2]], [[3, 1], [1, 4]]])
    for name, construct, target in [
        ("interpolated", _interpolated, "get_inverse_vandermonde_matrices_cached"),
        ("specialization", _specialized, "get_inverse_vandermonde_matrix"),
    ]:
        with case(path=name):
            with monkeypatch.context() as patch:
                # A singular modular system is the expected per-prime failure.
                patch.setattr(parallel, target, _raise(ZeroDivisionError("singular")))
                with pytest.raises(RuntimeError, match="8 consecutive primes"):
                    construct(pencil)
            with monkeypatch.context() as patch:
                # Anything else is a defect that must propagate unchanged.
                patch.setattr(parallel, target, _raise(_WorkerDefect("boom")))
                with pytest.raises(_WorkerDefect, match="boom"):
                    construct(pencil)


def test_isolated_prime_failures_still_reconstruct_exactly(
    bounded_crt: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrices = [[[1, 0], [0, 2]], [[3, 1], [1, 4]]]
    pencil = SymmetricMatrixPencil(matrices)
    x0, x1, z = sp.symbols("x1 x2 z")
    expected = sp.expand(
        sp.Matrix(matrices[0]) * x0 + sp.Matrix(matrices[1]) * x1
    ).det()
    expected_specialization = sp.Poly(
        (sp.Matrix(matrices[0]) * (z + 3) + sp.Matrix(matrices[1]) * (2 * z - 1)).det(),
        z,
    ).monic()
    for name, attribute, module in [
        ("interpolated", "_eval_prime_worker", multivariate),
        ("specialization", "_eval_diagonal_specialization_prime_worker", pencils),
    ]:
        with case(path=name):
            worker = getattr(module, attribute)
            calls = {"count": 0}

            def flaky(
                p: int, *args: Any, _worker: Any = worker, _calls: Any = calls
            ) -> Any:
                _calls["count"] += 1
                # Seven failures, then a success, repeating.
                if _calls["count"] % 8 != 0:
                    return p, None
                return _worker(p, *args)

            with monkeypatch.context() as patch:
                patch.setattr(module, attribute, flaky)
                if name == "interpolated":
                    result = _interpolated(pencil)
                    assert sp.expand(result.expr - expected) == 0
                else:
                    result = _specialized(pencil)
                    assert list(result.coeffs) == expected_specialization.all_coeffs()
