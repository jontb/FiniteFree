from typing import Any, Iterator

import numpy as np
import pytest
import scipy.linalg
import sympy as sp

from finitefree import gue_expected_poly, matrix_moments, sample_beta_hermite
from tests._cases import case


@pytest.fixture(autouse=True)
def preserve_numpy_rng() -> Iterator[None]:
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


class ControlledGenerator(np.random.Generator):
    """Return fixed normals and chi-square values, recording the requests."""

    def __init__(self, normals: Any = None, chi_squares: Any = None) -> None:
        super().__init__(np.random.PCG64(0))
        self.normals = normals
        self.chi_squares = chi_squares
        self.calls: list[tuple[str, Any]] = []

    def standard_normal(self, size: Any = None, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("standard_normal", size))
        values = np.ones(size) if self.normals is None else self.normals
        return np.array(values, dtype=float)

    def chisquare(self, df: Any, size: Any = None) -> Any:
        df = np.array(df, dtype=float)
        self.calls.append(("chisquare", df.copy()))
        # Default to the chi-square means, so entries equal their expectations.
        return df if self.chi_squares is None else np.array(self.chi_squares)


def dense(diagonal: Any, off_diagonal: Any) -> Any:
    return np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)


def test_controlled_construction_scale_and_order() -> None:
    normals = [0.5, -1.0, 2.0, 0.25]
    chi_squares = [9.0, 4.0, 1.0]
    for beta, scale in [(2.0, 1.0), (0.5, -3.0), (4, 0.25)]:
        with case(beta=beta, scale=scale):
            stream = ControlledGenerator(normals, chi_squares)
            diagonal, off = sample_beta_hermite(4, beta, scale, rng=stream)
            factor = scale / np.sqrt(beta * 4)
            np.testing.assert_allclose(
                diagonal, np.sqrt(2) * factor * np.array(normals), rtol=1e-15
            )
            np.testing.assert_allclose(off, factor * np.array([3.0, 2.0, 1.0]))
            assert [name for name, _ in stream.calls] == [
                "standard_normal",
                "chisquare",
            ]
            assert stream.calls[0][1] == 4
            np.testing.assert_array_equal(
                stream.calls[1][1], beta * np.array([3.0, 2.0, 1.0])
            )


def test_second_moment_identity_from_entry_expectations() -> None:
    # tr(H^2) is linear in each normal squared and each chi square; ones and
    # chi-square means therefore give E[tr(H^2)] exactly.
    for beta in [sp.Rational(1, 3), 1, 2, 4, sp.Rational(73, 10)]:
        for d in [1, 2, 3, 7]:
            with case(beta=beta, d=d):
                diagonal, off = sample_beta_hermite(
                    d, float(beta), rng=ControlledGenerator()
                )
                second = (np.sum(diagonal**2) + 2 * np.sum(off**2)) / d
                expected = 1 + (2 / beta - 1) / sp.Integer(d)
                assert second == pytest.approx(float(expected), rel=1e-14)


def test_expected_characteristic_polynomial_is_beta_independent() -> None:
    # det(x-H) is affine in each diagonal entry and each squared off-diagonal
    # entry, so its expectation is the determinant at those entry means.
    for beta in [0.5, 1.0, 2.0, 4.0, 7.3]:
        for d in range(1, 7):
            with case(beta=beta, d=d):
                diagonal, off = sample_beta_hermite(
                    d, beta, rng=ControlledGenerator(normals=np.zeros(d))
                )
                coefficients = np.poly(dense(diagonal, off))
                expected = [float(c) for c in gue_expected_poly(d).coeffs]
                np.testing.assert_allclose(coefficients, expected, rtol=0, atol=1e-13)


def test_one_dimensional_law_and_empty_off_diagonal() -> None:
    for beta in [0.5, 1.0, 4.0]:
        with case(beta=beta):
            stream = ControlledGenerator(normals=[1.5])
            diagonal, off = sample_beta_hermite(1, beta, 2.0, rng=stream)
            # One N(0, 2 scale^2 / beta) value: the unit normal times sqrt(8/beta).
            np.testing.assert_allclose(diagonal, [1.5 * np.sqrt(8 / beta)])
            assert off.shape == (0,) and off.dtype == np.float64
            assert [name for name, _ in stream.calls] == ["standard_normal"]


def test_noninteger_beta_and_eigenvalues_from_compact_output() -> None:
    rng = np.random.default_rng(43)
    for beta in [0.37, 2.5, 11.0]:
        with case(beta=beta):
            diagonal, off = sample_beta_hermite(6, beta, rng=rng)
            assert np.all(np.isfinite(diagonal)) and np.all(off > 0)
            eigenvalues = scipy.linalg.eigvalsh_tridiagonal(diagonal, off)
            np.testing.assert_allclose(
                matrix_moments(dense(diagonal, off), 4)[1:],
                [np.sum(eigenvalues**k) / 6 for k in range(1, 5)],
                rtol=1e-12,
                atol=1e-13,
            )


def test_rng_streams_and_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    np.random.seed(1729)
    legacy = sample_beta_hermite(5, 1.5)
    after = np.random.get_state()
    reference = np.random.RandomState(1729)
    normals = reference.standard_normal(5)
    chi_squares = reference.chisquare(1.5 * np.arange(4, 0, -1.0))
    factor = 1 / np.sqrt(1.5 * 5)
    np.testing.assert_array_equal(legacy[0], np.sqrt(2.0) * factor * normals)
    np.testing.assert_array_equal(legacy[1], factor * np.sqrt(chi_squares))
    assert str(after) == str(reference.get_state())

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Explicit Generator accessed the global RNG")

    monkeypatch.setattr(np.random, "standard_normal", forbidden)
    monkeypatch.setattr(np.random, "chisquare", forbidden)
    first = sample_beta_hermite(5, 1.5, rng=np.random.default_rng(8))
    second = sample_beta_hermite(5, 1.5, rng=np.random.default_rng(8))
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    for invalid in (7, np.random.RandomState(8), object()):
        with case(rng=type(invalid).__name__):
            with pytest.raises(TypeError, match="Generator or None"):
                sample_beta_hermite(5, 1.5, rng=invalid)  # type: ignore[arg-type]


def test_invalid_arguments_consume_no_randomness() -> None:
    invalid: list[tuple[dict[str, Any], str]] = [
        ({"d": 0}, "d must be a positive integer"),
        ({"d": 2.0}, "d must be a positive integer"),
        ({"d": True}, "d must be a positive integer"),
        ({"beta": 0}, "beta must be a finite positive"),
        ({"beta": -1.0}, "beta must be a finite positive"),
        ({"beta": np.inf}, "beta must be a finite positive"),
        ({"beta": np.nan}, "beta must be a finite positive"),
        ({"beta": True}, "beta must be a finite positive"),
        ({"beta": 1 + 0j}, "beta must be a finite positive"),
        ({"beta": "2"}, "beta must be a finite positive"),
        ({"scale": 0.0}, "scale must be a finite nonzero"),
        ({"scale": np.inf}, "scale must be a finite nonzero"),
        ({"scale": False}, "scale must be a finite nonzero"),
    ]
    for override, message in invalid:
        arguments: dict[str, Any] = {"d": 3, "beta": 2.0, "scale": 1.0, **override}
        with case(**override):
            stream = np.random.default_rng(5)
            before_stream = stream.bit_generator.state
            before_global = str(np.random.get_state())
            with pytest.raises(ValueError, match=message):
                sample_beta_hermite(**arguments, rng=stream)
            with pytest.raises(ValueError, match=message):
                sample_beta_hermite(**arguments)
            assert stream.bit_generator.state == before_stream
            assert str(np.random.get_state()) == before_global


def test_integer_index_types_ownership_and_overflow() -> None:
    first = sample_beta_hermite(
        np.int64(3), np.float32(2), rng=np.random.default_rng(1)
    )
    second = sample_beta_hermite(3, 2.0, rng=np.random.default_rng(1))
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert left.dtype == np.float64 and left.flags.writeable
        assert not np.shares_memory(left, right)
    assert sample_beta_hermite(2, sp.Rational(1, 2), rng=ControlledGenerator())[
        0
    ].shape == (2,)
    with pytest.raises(RuntimeError, match="finite float64 range"):
        sample_beta_hermite(1, 1.0, 1e308, rng=ControlledGenerator(normals=[10.0]))
