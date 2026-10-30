from typing import Any

import numpy as np
import pytest
import sympy as sp

from finitefree import (
    FiniteRTransform,
    RealRootedPolynomial,
    inverse_finite_r,
    matrix_finite_cumulants,
    matrix_moments,
    root_moments,
    sample_goe,
    sample_gse,
    sample_gue,
)
from finitefree.streaming import _sample_observations
from tests._cases import case
from tests._cumulants import logarithm_reference


def trace_moments(matrix: Any, order: int, denominator: int) -> Any:
    """Independent oracle: direct normalized matrix-power traces."""
    sample = np.asarray(matrix, dtype=np.complex128)
    return np.array(
        [1.0]
        + [
            np.trace(np.linalg.matrix_power(sample, k)).real / denominator
            for k in range(1, order + 1)
        ]
    )


def test_complex_hermitian_matches_exact_traces() -> None:
    exact = sp.Matrix([[2, sp.I], [-sp.I, 2]])
    numeric = np.array([[2, 1j], [-1j, 2]])
    for dimension in [2, 4]:
        with case(d=dimension):
            expected = [1.0] + [
                float((exact**k).trace() / dimension) for k in range(1, 5)
            ]
            result = matrix_moments(numeric, 4, d=None if dimension == 2 else 4)
            np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(matrix_moments(numeric, 4), [1, 2, 5, 14, 41])


def test_paired_representation_uses_one_eigenvalue_per_pair() -> None:
    matrix = np.diag([1.0, 1.0, 3.0, 3.0])
    for dimension, denominator, expected in [
        (None, 4, [1, 2, 5, 14, 41]),
        (4, 8, [1, 1, 2.5, 7, 20.5]),
    ]:
        with case(d=dimension):
            result = matrix_moments(matrix, 4, d=dimension, paired=True)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_allclose(
                result, trace_moments(matrix, 4, denominator), rtol=1e-15
            )
    # Without explicit pairing all four physical eigenvalues are counted.
    np.testing.assert_array_equal(
        matrix_moments(matrix, 2), [1, 2, 5]
    )  # (1+1+3+3)/4, (1+1+9+9)/4
    np.testing.assert_array_equal(matrix_moments(matrix, 2, d=8), [1, 1, 2.5])


def test_seeded_samples_match_direct_traces() -> None:
    rng = np.random.default_rng(20261029)
    for name, matrix, paired in [
        ("goe", sample_goe(5, rng=rng), False),
        ("gue", sample_gue(4, scale=3.0, rng=rng), False),
        ("gse", sample_gse(3, rng=rng), True),
    ]:
        size = matrix.shape[0]
        effective = size // 2 if paired else size
        for dimension in [effective, effective + 2]:
            with case(ensemble=name, d=dimension):
                denominator = dimension * (2 if paired else 1)
                np.testing.assert_allclose(
                    matrix_moments(matrix, 7, d=dimension, paired=paired),
                    trace_moments(matrix, 7, denominator),
                    rtol=1e-11,
                    atol=1e-12,
                )


def test_diagonal_spectra_match_exact_root_moments() -> None:
    spectra: list[list[float]] = [[1, 3], [-2, -2, 0, 0.5], [1e-3, 7, 7, 7], [0.0]]
    for spectrum in spectra:
        for extra in [0, 2]:
            dimension = len(spectrum) + extra
            with case(spectrum=spectrum, d=dimension):
                exact = root_moments(
                    RealRootedPolynomial.from_roots(spectrum), 6, d=dimension
                )
                result = matrix_moments(np.diag(spectrum), 6, d=dimension)
                np.testing.assert_allclose(
                    result, [float(value) for value in exact], rtol=1e-14, atol=0
                )


def test_scale_acts_on_the_sampled_matrix() -> None:
    matrix = sample_gue(4, rng=np.random.default_rng(5))
    base = matrix_moments(matrix, 5)
    for scale in [0.25, -3.0]:
        with case(scale=scale):
            np.testing.assert_allclose(
                matrix_moments(scale * matrix, 5),
                base * scale ** np.arange(6),
                rtol=1e-12,
                atol=1e-14,
            )


def test_mean_spectral_moments_differ_from_mean_polynomial() -> None:
    draws = [np.diag([1.0, 3.0]), np.diag([3.0, 5.0])]
    mean_moments = np.mean([matrix_moments(draw, 2) for draw in draws], axis=0)
    np.testing.assert_array_equal(mean_moments, [1, 3, 11])
    mean_poly = RealRootedPolynomial([1, -6, 9])  # mean of (x-1)(x-3), (x-3)(x-5)
    assert root_moments(mean_poly, 2) == [1, 3, 9]
    per_draw = [RealRootedPolynomial.from_roots(r) for r in ([1, 3], [3, 5])]
    assert [FiniteRTransform(p, order=2) for p in per_draw] == [[2, 2], [4, 2]]
    assert FiniteRTransform(mean_poly, order=2) == [3, 0]


def test_representable_moments_despite_coefficient_overflow() -> None:
    matrix = np.diag(np.full(20, 1e40))
    with pytest.raises(ValueError, match="characteristic coefficients"):
        _sample_observations(matrix, 20)
    np.testing.assert_allclose(matrix_moments(matrix, 2), [1, 1e40, 1e80], rtol=1e-14)


def test_nonfinite_moments_raise() -> None:
    for spectrum, order in [([1e200, 1.0], 2), ([1e308, 1e308], 1)]:
        with case(spectrum=spectrum, order=order):
            with pytest.raises(RuntimeError, match="finite float64 range"):
                matrix_moments(np.diag(spectrum), order)


def test_invalid_matrices_and_indices() -> None:
    hermitian = np.diag([1.0, 2.0])
    invalid: list[tuple[Any, str]] = [
        (np.zeros((0, 0)), "nonempty square"),
        (np.zeros((2, 3)), "nonempty square"),
        (np.array([1.0, 2.0]), "nonempty square"),
        (np.zeros((2, 2, 2)), "nonempty square"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite entries"),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "numerically Hermitian"),
        (np.array([[1.0, 1j], [1j, 1.0]]), "numerically Hermitian"),
    ]
    for matrix, message in invalid:
        with case(matrix=matrix.tolist()):
            with pytest.raises(ValueError, match=message):
                matrix_moments(matrix, 2)
    for order in [-1, 2.5, True, "1"]:
        with case(order=order):
            with pytest.raises(ValueError, match="order must be a nonnegative"):
                matrix_moments(hermitian, order)  # type: ignore[arg-type]
    for dimension in [0, 1.0, True]:
        with case(d=dimension):
            with pytest.raises(ValueError, match="d must be a positive"):
                matrix_moments(hermitian, 1, d=dimension)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effective matrix dimension"):
        matrix_moments(hermitian, 1, d=1)
    with pytest.raises(ValueError, match="effective matrix dimension"):
        matrix_moments(np.diag([1.0, 1.0, 2.0, 2.0]), 1, d=1, paired=True)
    with pytest.raises(ValueError, match="paired must be a boolean"):
        matrix_moments(hermitian, 1, paired=1)  # type: ignore[arg-type]


def test_paired_mismatch_is_rejected() -> None:
    for matrix in [np.diag([1.0, 2.0, 3.0, 4.0]), np.diag([1.0, 1.0, 2.0])]:
        with case(diagonal=np.diag(matrix).tolist()):
            with pytest.raises(ValueError, match="adjacent eigenvalue pairs"):
                matrix_moments(matrix, 2, paired=True)


def test_noncontiguous_input_is_not_mutated_and_output_is_owned() -> None:
    rng = np.random.default_rng(11)
    large = sample_gue(6, rng=rng)
    views = [large[::2, ::2], np.asfortranarray(large.real), large.T.conj()]
    for index, view in enumerate(views):
        with case(view=index):
            before = view.copy()
            expected = trace_moments(view, 4, view.shape[0])
            first = matrix_moments(view, 4)
            np.testing.assert_array_equal(view, before)
            np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-13)
            assert first.dtype == np.float64 and first.flags.writeable
            first[:] = 0
            second = matrix_moments(view, 4)
            np.testing.assert_allclose(second, expected, rtol=1e-12, atol=1e-13)
            assert not np.shares_memory(first, second)


def test_seeded_gue_example_uses_the_same_draws() -> None:
    rng = np.random.default_rng(862)
    draws = [sample_gue(4, rng=rng) for _ in range(16)]
    routed = np.array([matrix_moments(draw, 4) for draw in draws])
    traced = np.array([trace_moments(draw, 4, 4) for draw in draws])
    # Numerical agreement on each draw is deterministic; sampling spread is not.
    np.testing.assert_allclose(routed, traced, rtol=1e-12, atol=1e-13)
    standard_error = routed.std(axis=0, ddof=1) / np.sqrt(len(draws))
    assert routed[:, 0].tolist() == [1.0] * 16
    assert standard_error[0] == 0 and np.all(standard_error[1:] > 0)


def exact_cumulants(matrix: Any, d: int) -> list[Any]:
    """Independent oracle: exact determinant coefficients, then a formal logarithm."""
    rows = np.asarray(matrix, dtype=np.complex128)
    exact = sp.Matrix(
        [
            [
                sp.Rational(float(v.real)) + sp.I * sp.Rational(float(v.imag))
                for v in row
            ]
            for row in rows
        ]
    )
    coefficients = [sp.expand(c) for c in exact.charpoly().all_coeffs()]
    assert all(sp.im(c) == 0 for c in coefficients)
    p = RealRootedPolynomial([sp.re(c) for c in coefficients], assume_real_rooted=True)
    return logarithm_reference(p, d, d)


def test_matrix_cumulants_match_exact_determinant_references() -> None:
    rng = np.random.default_rng(42)
    matrices = [
        np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]),
        np.array([[2, 1 + 1j], [1 - 1j, 3]]),
        sample_goe(4, rng=rng),
        sample_gue(4, scale=2.0, rng=rng),
    ]
    for index, matrix in enumerate(matrices):
        n = matrix.shape[0]
        for dimension in [n, n + 2]:
            with case(matrix=index, d=dimension):
                expected = [float(v) for v in exact_cumulants(matrix, n)]
                if dimension > n:
                    padded = np.zeros((dimension, dimension), dtype=matrix.dtype)
                    padded[:n, :n] = matrix
                    expected = [float(v) for v in exact_cumulants(padded, dimension)]
                np.testing.assert_allclose(
                    matrix_finite_cumulants(matrix, d=dimension),
                    expected,
                    rtol=1e-10,
                    atol=1e-12,
                )


def test_matrix_cumulant_fixed_values_and_pairing() -> None:
    np.testing.assert_allclose(matrix_finite_cumulants([[2, 1j], [-1j, 2]]), [2, 2])
    paired = np.diag([1.0, 1.0, 3.0, 3.0])
    np.testing.assert_allclose(matrix_finite_cumulants(paired, paired=True), [2, 2])
    np.testing.assert_allclose(
        matrix_finite_cumulants(paired, d=4, paired=True), [1, 2, 4, 8]
    )
    # Without explicit pairing all four eigenvalues count: roots [1,1,3,3].
    expected = logarithm_reference(RealRootedPolynomial.from_roots([1, 1, 3, 3]), 4, 4)
    np.testing.assert_allclose(
        matrix_finite_cumulants(paired), [float(v) for v in expected], atol=1e-13
    )
    gse = sample_gse(3, rng=np.random.default_rng(9))
    half = np.linalg.eigvalsh(gse)[::2]
    np.testing.assert_allclose(
        matrix_finite_cumulants(gse, paired=True),
        matrix_finite_cumulants(np.diag(half)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_matrix_cumulants_center_large_offsets() -> None:
    offset = 1e8
    result = matrix_finite_cumulants(np.diag([offset + 1, offset + 3, offset + 8]))
    exact = logarithm_reference(RealRootedPolynomial.from_roots([1, 3, 8]), 3, 3)
    assert result[0] == offset + 4
    np.testing.assert_allclose(result[1:], [float(v) for v in exact[1:]], rtol=1e-9)


def test_cumulant_targets_and_empirical_round_trip() -> None:
    draws = [np.diag([1.0, 3.0]), np.diag([3.0, 5.0])]
    per_draw = [matrix_finite_cumulants(draw) for draw in draws]
    np.testing.assert_allclose(per_draw, [[2, 2], [4, 2]])
    mean_cumulants = np.mean(per_draw, axis=0)
    np.testing.assert_allclose(
        inverse_finite_r(mean_cumulants, numerical=True), [1, -6, 8]
    )  # (x-2)(x-4)
    assert FiniteRTransform(RealRootedPolynomial([1, -6, 9]), order=2) == [3, 0]

    rng = np.random.default_rng(862)
    for draw in [sample_gue(5, rng=rng) for _ in range(4)]:
        reconstructed = inverse_finite_r(matrix_finite_cumulants(draw), numerical=True)
        np.testing.assert_allclose(
            reconstructed,
            np.poly(np.linalg.eigvalsh(draw)),
            rtol=1e-10,
            atol=1e-12,
        )


def test_matrix_cumulant_validation_overflow_and_ownership() -> None:
    with pytest.raises(ValueError, match="nonempty square"):
        matrix_finite_cumulants(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="adjacent eigenvalue pairs"):
        matrix_finite_cumulants(np.diag([1.0, 2.0]), paired=True)
    with pytest.raises(ValueError, match="effective matrix dimension"):
        matrix_finite_cumulants(np.diag([1.0, 2.0]), d=1)
    with pytest.raises(RuntimeError, match="finite float64 range"):
        matrix_finite_cumulants(np.diag([1e200, -1e200]))
    view = sample_gue(6, rng=np.random.default_rng(3))[::2, ::2]
    before = view.copy()
    first = matrix_finite_cumulants(view)
    np.testing.assert_array_equal(view, before)
    first[:] = 0
    assert np.any(matrix_finite_cumulants(view) != 0)
