import math
from typing import Any

import numpy as np
import pytest

from finitefree import (
    EmpiricalComparison,
    RealRootedPolynomial,
    StreamingEmpiricalComparison,
)
from tests._cases import case


def test_streaming_matches_retained_coefficients_counts_and_update_order() -> None:
    eigenvalues = np.random.default_rng(22).uniform(-2, 2, size=(137, 4))
    matrices = [np.diag(row) for row in eigenvalues]
    polynomial = RealRootedPolynomial.from_roots([-1, 0, 1, 2])
    retained = EmpiricalComparison(polynomial, len(matrices), iter(matrices).__next__)
    for observations in [matrices, matrices[::-1]]:
        with case(reversed=observations is not matrices):
            intervals = [(-1.0, 1.0), (0.0, 2.0), (-3.0, -2.0)]
            stream = StreamingEmpiricalComparison(polynomial, intervals)
            stream.extend(iter(observations[:33]).__next__, 33)
            for matrix in observations[33:]:
                stream.update(matrix)
            np.testing.assert_allclose(
                stream.coefficient_mean,
                retained.char_poly_coeffs.mean(axis=0),
                atol=2e-15,
            )
            np.testing.assert_allclose(
                stream.coefficient_std,
                retained.char_poly_coeffs.std(axis=0, ddof=1),
                atol=2e-15,
            )
            counts = np.array(
                [
                    [np.count_nonzero((row >= a) & (row <= b)) for a, b in intervals]
                    for row in eigenvalues
                ]
            )
            np.testing.assert_allclose(stream.count_mean, counts.mean(axis=0))
            np.testing.assert_allclose(
                stream.count_variance, counts.var(axis=0, ddof=1)
            )
            for j in range(3):
                np.testing.assert_array_equal(
                    stream.count_histograms[j], np.bincount(counts[:, j], minlength=5)
                )
            assert stream.samples == len(matrices)
            for alpha in [0.05, 0.2]:
                assert stream.verify_coefficients(
                    alpha
                ) == retained.verify_coefficients(alpha)
            # Public snapshots cannot alter subsequent updates/results.
            stream.coefficient_mean[:] = 999
            stream.count_histograms[:] = 999
            stream.intervals[:] = 999
            assert stream.count_histograms.sum() == 3 * stream.samples
            assert not hasattr(stream, "eigenvalues")
            assert not hasattr(stream, "char_poly_coeffs")
            sizes = {
                k: v.shape for k, v in vars(stream).items() if isinstance(v, np.ndarray)
            }
            stream.extend(lambda: np.eye(4), 1000)
            assert sizes == {
                k: v.shape for k, v in vars(stream).items() if isinstance(v, np.ndarray)
            }


def test_streaming_closed_form_t_bands_and_extreme_dynamic_rescaling() -> None:
    for scale in [1e-250, 1.0, 1e250]:
        for alpha in [0.05, 0.2]:
            u = alpha / 4
            width = (1 - 2 * u) / math.sqrt(2 * u * (1 - u)) / math.sqrt(3)
            for factor in [0.99, 1.01]:
                with case(scale=scale, alpha=alpha, factor=factor):
                    stream = StreamingEmpiricalComparison(
                        RealRootedPolynomial.from_roots([factor * width * scale])
                    )
                    stream.extend(
                        iter(
                            [
                                np.array([[-scale]]),
                                np.array([[0.0]]),
                                np.array([[scale]]),
                            ]
                        ).__next__,
                        3,
                    )
                    assert stream.verify_coefficients(alpha, rtol=0) == (factor < 1)
            deterministic = StreamingEmpiricalComparison(
                RealRootedPolynomial.from_roots([0])
            )

            def constant_sample(scale: float = scale) -> Any:
                return np.array([[scale]])

            deterministic.extend(constant_sample, 3)
            assert not deterministic.verify_coefficients()
            assert deterministic.verify_coefficients(atol=scale)
    observations = [1e-250, -1e-250, 1e250, -1e250, 0.0]
    stream = StreamingEmpiricalComparison(RealRootedPolynomial.from_roots([0]))
    stream.extend(iter([np.array([[x]]) for x in observations]).__next__, 5)
    assert stream.verify_coefficients()
    assert stream.coefficient_std[1] == pytest.approx(math.sqrt(0.5) * 1e250)
    assert stream.count_histograms.shape == (0, 2)


def test_streaming_rejects_invalid_samples_atomically_and_handles_boundaries() -> None:
    stream = StreamingEmpiricalComparison(
        RealRootedPolynomial.from_roots([1, 3]), [(1, 3), (1, 1), (3, 3)]
    )
    for value in [np.array([[2, 1j], [-1j, 2]]), np.diag([1, 1, 3, 3])]:
        stream.update(value)
    assert stream.verify_coefficients()
    # Diagonal samples exercise exact closed interval boundaries.
    exact = StreamingEmpiricalComparison(
        RealRootedPolynomial.from_roots([1, 3]), [(1, 3), (1, 1), (3, 3)]
    )
    exact.extend(lambda: np.diag([1, 3]), 3)
    np.testing.assert_array_equal(exact.count_mean, [2, 1, 1])
    np.testing.assert_array_equal(exact.count_variance, [0, 0, 0])
    matrix: Any
    for matrix in [
        np.eye(3),
        np.eye(2) * 1j,
        [[0, 1], [0, 0]],
        np.full((2, 2), np.nan),
        np.diag([1, 2, 3, 4]),
        np.eye(2) * 1e200,
    ]:
        before = stream.coefficient_mean
        with case(matrix=matrix), pytest.raises(ValueError):
            stream.update(matrix)
        assert stream.samples == 2
        np.testing.assert_array_equal(stream.coefficient_mean, before)
    intervals: Any
    for intervals in [[(1, 0)], [(0, np.inf)], [(1j, 2)], [(0, 1, 2)], [1, 2]]:
        with case(intervals=intervals), pytest.raises(ValueError, match="Intervals"):
            StreamingEmpiricalComparison(RealRootedPolynomial([1]), intervals)
    empty = StreamingEmpiricalComparison(RealRootedPolynomial([1]))
    with pytest.raises(ValueError):
        empty.verify_coefficients()
    n: Any
    for n in [-1, 1.5, True]:
        with case(n=n), pytest.raises(ValueError):
            empty.extend(lambda: np.empty((0, 0)), n)
    empty.extend(lambda: np.empty((0, 0)), 0)
    assert empty.samples == 0
    empty.extend(lambda: np.empty((0, 0)), 3)
    assert empty.verify_coefficients()
    np.testing.assert_array_equal(empty.coefficient_mean, [1])
    args: Any
    for args in [{"alpha": 0}, {"alpha": 5e-324}, {"rtol": -1}, {"atol": np.inf}]:
        with case(args=args), pytest.raises(ValueError):
            empty.verify_coefficients(**args)
