from typing import Any, Iterator

import numpy as np
import pytest
import scipy.linalg
import sympy as sp

from finitefree import matrix_moments, sample_beta_laguerre, wishart_expected_poly
from tests._cases import case


@pytest.fixture(autouse=True)
def preserve_numpy_rng() -> Iterator[None]:
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


class ControlledGenerator(np.random.Generator):
    """Return fixed chi-square values (default: their means), recording requests."""

    def __init__(self, *values: Any) -> None:
        super().__init__(np.random.PCG64(0))
        self.values = list(values)
        self.calls: list[Any] = []

    def chisquare(self, df: Any, size: Any = None) -> Any:
        df = np.array(df, dtype=float)
        self.calls.append(df.copy())
        return np.array(self.values.pop(0), dtype=float) if self.values else df

    def standard_normal(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("beta-Laguerre sampling must not draw normals")


def dense_factor(diagonal: Any, subdiagonal: Any) -> Any:
    return np.diag(diagonal) + np.diag(subdiagonal, -1)


def laguerre_reference(d: int, n: Any) -> list[float]:
    """Independent oracle: n^-d d! (-1)^d L_d^(n-d)(n x) from SymPy."""
    x = sp.Symbol("x")
    n = sp.nsimplify(n)
    poly = n**-d * sp.factorial(d) * (-1) ** d * sp.assoc_laguerre(d, n - d, n * x)
    return [float(c) for c in sp.Poly(sp.expand(poly), x).all_coeffs()]


def test_controlled_factor_scale_order_and_gram_assembly() -> None:
    diagonal_draws = [4.0, 9.0, 1.0]
    subdiagonal_draws = [16.0, 0.25]
    for n, beta, scale in [(3.0, 2.0, 1.0), (4.5, 0.5, 3.0), (2.25, 4, 0.5)]:
        with case(n=n, beta=beta, scale=scale):
            stream = ControlledGenerator(diagonal_draws, subdiagonal_draws)
            diagonal, subdiagonal = sample_beta_laguerre(3, n, beta, scale, rng=stream)
            factor = np.sqrt(scale / (beta * n))
            np.testing.assert_allclose(diagonal, factor * np.array([2.0, 3.0, 1.0]))
            np.testing.assert_allclose(subdiagonal, factor * np.array([4.0, 0.5]))
            np.testing.assert_allclose(stream.calls[0], beta * (n - np.arange(3)))
            np.testing.assert_allclose(stream.calls[1], beta * np.array([2.0, 1.0]))
            assert len(stream.calls) == 2
            # The dense product B B^T is the positive tridiagonal Gram matrix.
            gram = (
                dense_factor(diagonal, subdiagonal)
                @ dense_factor(diagonal, subdiagonal).T
            )
            expected_diagonal = diagonal**2 + np.concatenate([[0], subdiagonal**2])
            np.testing.assert_allclose(np.diag(gram), expected_diagonal)
            np.testing.assert_allclose(np.diag(gram, -1), diagonal[:-1] * subdiagonal)
            assert np.count_nonzero(np.triu(gram, 2)) == 0


def test_mean_trace_from_entry_expectations() -> None:
    # tr(W) is linear in the chi squares; their means give E[tr W] = scale d.
    for beta in [0.5, 1.0, 2.0, 4.0, 7.3]:
        for d, n in [(1, 0.3), (2, 2.0), (4, 3.5), (6, 11.0)]:
            for scale in [1.0, 2.5]:
                with case(beta=beta, d=d, n=n, scale=scale):
                    diagonal, subdiagonal = sample_beta_laguerre(
                        d, n, beta, scale, rng=ControlledGenerator()
                    )
                    trace = np.sum(diagonal**2) + np.sum(subdiagonal**2)
                    assert trace == pytest.approx(scale * d, rel=1e-14)


def test_expected_characteristic_polynomial_is_generalized_laguerre() -> None:
    # det(x - B B^T) is affine in each squared entry of B, so entry means give
    # its expectation; the result must not depend on beta.
    for d, n in [(1, 0.4), (2, 2), (3, 3.5), (4, 3.1), (5, 12.25), (6, 6)]:
        reference = laguerre_reference(d, n)
        for beta in [0.5, 1.0, 2.0, 4.0, 7.3]:
            with case(d=d, n=n, beta=beta):
                diagonal, subdiagonal = sample_beta_laguerre(
                    d, n, beta, rng=ControlledGenerator()
                )
                factor = dense_factor(diagonal, subdiagonal)
                np.testing.assert_allclose(
                    np.poly(factor @ factor.T), reference, rtol=0, atol=1e-12
                )
        if float(n).is_integer():
            with case(d=d, n=n, reference="wishart_expected_poly"):
                np.testing.assert_allclose(
                    [float(c) for c in wishart_expected_poly(d, int(n)).coeffs],
                    reference,
                    rtol=0,
                    atol=1e-13,
                )


def test_one_dimensional_gamma_law_and_empty_subdiagonal() -> None:
    for n, beta, scale in [(0.3, 1.0, 1.0), (5.0, 0.5, 2.0), (2.0, 4.0, 0.25)]:
        with case(n=n, beta=beta, scale=scale):
            stream = ControlledGenerator([7.0])
            diagonal, subdiagonal = sample_beta_laguerre(1, n, beta, scale, rng=stream)
            # W = scale * chi2_{beta n} / (beta n): a gamma(beta n/2, 2 scale/(beta n)) law.
            assert diagonal[0] ** 2 == pytest.approx(scale * 7.0 / (beta * n))
            np.testing.assert_allclose(stream.calls[0], [beta * n])
            assert subdiagonal.shape == (0,) and subdiagonal.dtype == np.float64
            assert len(stream.calls) == 1


def test_noninteger_shapes_near_boundary_and_eigenvalue_routes() -> None:
    rng = np.random.default_rng(44)
    for d, n, beta in [(4, 3.000001, 2.0), (5, 6.75, 0.37), (3, 2.5, 11.0)]:
        with case(d=d, n=n, beta=beta):
            diagonal, subdiagonal = sample_beta_laguerre(d, n, beta, rng=rng)
            assert np.all(diagonal >= 0) and np.all(subdiagonal > 0)
            factor = dense_factor(diagonal, subdiagonal)
            squared_singular = np.sort(scipy.linalg.svdvals(factor) ** 2)
            gram_diagonal = diagonal**2 + np.concatenate([[0], subdiagonal**2])
            gram_eigenvalues = scipy.linalg.eigvalsh_tridiagonal(
                gram_diagonal, diagonal[:-1] * subdiagonal
            )
            scale = np.max(gram_eigenvalues)
            np.testing.assert_allclose(
                squared_singular, gram_eigenvalues, rtol=0, atol=1e-12 * scale
            )
            np.testing.assert_allclose(
                matrix_moments(factor @ factor.T, 3)[1:],
                [np.mean(gram_eigenvalues**k) for k in range(1, 4)],
                rtol=1e-11,
            )


def test_rng_streams_and_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    np.random.seed(1729)
    legacy = sample_beta_laguerre(4, 5.5, 1.5)
    after = np.random.get_state()
    reference = np.random.RandomState(1729)
    diagonal_squares = reference.chisquare(1.5 * (5.5 - np.arange(4.0)))
    subdiagonal_squares = reference.chisquare(1.5 * np.arange(3, 0, -1.0))
    factor = np.sqrt(1 / (1.5 * 5.5))
    np.testing.assert_array_equal(legacy[0], factor * np.sqrt(diagonal_squares))
    np.testing.assert_array_equal(legacy[1], factor * np.sqrt(subdiagonal_squares))
    assert str(after) == str(reference.get_state())

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Explicit Generator accessed the global RNG")

    monkeypatch.setattr(np.random, "chisquare", forbidden)
    first = sample_beta_laguerre(4, 5.5, 1.5, rng=np.random.default_rng(8))
    second = sample_beta_laguerre(4, 5.5, 1.5, rng=np.random.default_rng(8))
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    for invalid in (7, np.random.RandomState(8), object()):
        with case(rng=type(invalid).__name__):
            with pytest.raises(TypeError, match="Generator or None"):
                sample_beta_laguerre(4, 5.5, 1.5, rng=invalid)  # type: ignore[arg-type]


def test_invalid_arguments_consume_no_randomness() -> None:
    invalid: list[tuple[dict[str, Any], str]] = [
        ({"d": 0}, "d must be a positive integer"),
        ({"d": 2.0}, "d must be a positive integer"),
        ({"d": True}, "d must be a positive integer"),
        ({"n": 2.0}, "n must exceed d - 1 = 2"),
        ({"n": 1.5}, "n must exceed d - 1 = 2"),
        ({"n": 0}, "n must be a finite positive"),
        ({"n": np.inf}, "n must be a finite positive"),
        ({"n": True}, "n must be a finite positive"),
        ({"n": 4 + 0j}, "n must be a finite positive"),
        ({"beta": 0}, "beta must be a finite positive"),
        ({"beta": -1.0}, "beta must be a finite positive"),
        ({"beta": np.nan}, "beta must be a finite positive"),
        ({"beta": "2"}, "beta must be a finite positive"),
        ({"scale": 0.0}, "scale must be a finite positive"),
        ({"scale": -2.0}, "scale must be a finite positive"),
        ({"scale": np.inf}, "scale must be a finite positive"),
        ({"scale": False}, "scale must be a finite positive"),
    ]
    for override, message in invalid:
        arguments: dict[str, Any] = {"d": 3, "n": 4.0, "beta": 2.0, **override}
        with case(**override):
            stream = np.random.default_rng(5)
            before_stream = stream.bit_generator.state
            before_global = str(np.random.get_state())
            with pytest.raises(ValueError, match=message):
                sample_beta_laguerre(**arguments, rng=stream)
            with pytest.raises(ValueError, match=message):
                sample_beta_laguerre(**arguments)
            assert stream.bit_generator.state == before_stream
            assert str(np.random.get_state()) == before_global


def test_index_types_ownership_and_nonfinite_entries() -> None:
    first = sample_beta_laguerre(
        np.int64(3), np.float32(4.5), sp.Rational(2), rng=np.random.default_rng(1)
    )
    second = sample_beta_laguerre(3, 4.5, 2.0, rng=np.random.default_rng(1))
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert left.dtype == np.float64 and left.flags.writeable
        assert not np.shares_memory(left, right)
    with pytest.raises(RuntimeError, match="finite float64 range"):
        sample_beta_laguerre(1, 2.0, 1.0, rng=ControlledGenerator([np.inf]))
