from typing import Any, Callable, Iterator

import numpy as np
import pytest

from finitefree import (
    EmpiricalComparison,
    RealRootedPolynomial,
    gue_expected_poly,
    sample_goe,
    sample_gse,
    sample_gue,
    sample_haar_orthogonal,
    sample_haar_symplectic,
    sample_haar_unitary,
    sample_wishart,
    wishart_expected_poly,
)
from finitefree.convolutions import symmetric_additive
from tests._cases import case


@pytest.fixture(autouse=True)
def preserve_numpy_rng() -> Iterator[None]:
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def symplectic_form(d: int) -> Any:
    return np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])


def assert_hermitian_structure(matrix: Any, d: int, beta: int) -> None:
    size = 2 * d if beta == 4 else d
    assert matrix.shape == (size, size)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0, atol=1e-12)
    if beta == 1:
        assert np.isrealobj(matrix)
    elif beta == 2:
        tolerance = 8 * size * np.finfo(float).eps * np.max(np.abs(matrix))
        np.testing.assert_allclose(np.imag(np.diag(matrix)), 0, rtol=0, atol=tolerance)
    else:
        j = symplectic_form(d)
        np.testing.assert_allclose(matrix, j @ matrix.conj() @ j.T, rtol=0, atol=1e-12)


def assert_coefficient_law(samples: Any, expected: Any) -> None:
    assert samples.ndim == 2 and samples.shape[1:] == expected.shape
    assert np.isfinite(samples).all()
    mean = samples.mean(axis=0)
    sem = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    # Preserve the five-SEM check and also test deterministic coefficients.
    tolerance = np.where(sem > 1e-10, 5 * sem, 1e-10)
    np.testing.assert_array_less(np.abs(mean - expected), tolerance)


def test_gaussian_sampler_structure() -> None:
    for beta, d, sampler in (
        (1, 5, sample_goe),
        (2, 5, sample_gue),
        (4, 4, sample_gse),
    ):
        with case(beta=beta, dimension=d):
            assert_hermitian_structure(sampler(d), d, beta)


def test_wishart_sampler_structure_and_beta_validation() -> None:
    # Complex GEMM can leave a roundoff-sized imaginary diagonal across BLAS builds.
    rounded = np.diag(np.array([1 + 1.8e-17j, 2 - 1.2e-17j, 3 + 5e-19j]))
    assert_hermitian_structure(rounded, 3, 2)
    corrupted = rounded.copy()
    corrupted[0, 0] += 1e-13j
    with pytest.raises(AssertionError):
        assert_hermitian_structure(corrupted, 3, 2)
    for beta in (1, 2, 4):
        with case(beta=beta):
            matrix = sample_wishart(3, 5, beta=beta)
            assert_hermitian_structure(matrix, 3, beta)
            assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    with pytest.raises(ValueError, match="beta must be 1, 2, or 4"):
        sample_wishart(d=3, n=5, beta=3)


def test_haar_sampler_structure() -> None:
    for beta, sampler in (
        (1, sample_haar_orthogonal),
        (2, sample_haar_unitary),
        (4, sample_haar_symplectic),
    ):
        with case(beta=beta):
            matrix = sampler(4)
            size = 8 if beta == 4 else 4
            assert matrix.shape == (size, size)
            np.testing.assert_allclose(
                matrix @ matrix.conj().T, np.eye(size), atol=1e-12
            )
            if beta == 1:
                assert np.isrealobj(matrix)
            if beta == 4:
                j = symplectic_form(4)
                np.testing.assert_allclose(matrix.T @ j @ matrix, j, atol=1e-12)


def test_empirical_expected_polynomials() -> None:
    for name, d, polynomial, generator in (
        ("gue", 4, gue_expected_poly(4), lambda: sample_gue(4)),
        (
            "wishart",
            3,
            wishart_expected_poly(3, 6, beta=2),
            lambda: sample_wishart(3, 6, beta=2),
        ),
    ):
        with case(ensemble=name):
            np.random.seed(42)
            assert polynomial.degree == d
            comparison = EmpiricalComparison(polynomial, 1000, generator)
            assert comparison.verify_coefficients(alpha=1e-6)


def test_empirical_conjugation_coefficient_laws() -> None:
    for beta, sampler in (
        (2, sample_haar_unitary),
        (1, sample_haar_orthogonal),
        (4, sample_haar_symplectic),
    ):
        with case(beta=beta):
            np.random.seed(42)
            if beta == 4:
                a, b, trials = (
                    np.array([1.5, 2.5, 3.5]),
                    np.array([-1.0, 0.0, 1.0]),
                    2000,
                )
            else:
                a, b, trials = (
                    np.array([1.0, 2.0, 3.0, 4.0]),
                    np.array([-2.0, -1.0, 1.0, 2.0]),
                    3000,
                )
            d = len(a)
            p, q = (
                RealRootedPolynomial(np.poly(roots), assume_real_rooted=True)
                for roots in (a, b)
            )
            expected = np.asarray(symmetric_additive(p, q, d).coeffs, dtype=float)
            left, right = np.diag(a), np.diag(b)
            if beta == 4:
                left, right = np.kron(np.eye(2), left), np.kron(np.eye(2), right)
            coefficients = []
            for _ in range(trials):
                unitary = sampler(d)
                matrix = left + unitary @ right @ unitary.conj().T
                if beta == 4:
                    roots = np.sort(np.linalg.eigvalsh(matrix))
                    np.testing.assert_allclose(roots[::2], roots[1::2], atol=1e-5)
                    sample = np.poly(roots[::2])
                else:
                    sample = np.poly(matrix)
                coefficients.append(np.real(sample))
            assert_coefficient_law(np.asarray(coefficients), expected)


def test_sampler_rng_default_stream_and_explicit_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samplers: list[
        tuple[
            Callable[..., Any],
            tuple[int, ...],
            dict[str, int],
            list[tuple[int, int]],
            int,
        ]
    ] = [
        (sample_goe, (3,), {}, [(3, 3)], 1),
        (sample_gue, (3,), {}, [(3, 3)] * 2, 2),
        (sample_gse, (3,), {}, [(6, 6)] * 2, 4),
        *[
            (sample_wishart, (3, 5), {"beta": beta}, [(3, 5)] * draws, beta)
            for beta, draws in ((1, 1), (2, 2), (4, 4))
        ],
        (sample_haar_orthogonal, (3,), {}, [(3, 3)], 1),
        (sample_haar_unitary, (3,), {}, [(3, 3)] * 2, 2),
        (sample_haar_symplectic, (3,), {}, [(3, 3)] * 4, 4),
    ]

    def assert_state_equal(left: Any, right: Any) -> None:
        assert left[0] == right[0]
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2:] == right[2:]

    for sampler, args, keywords, draw_shapes, beta in samplers:
        with case(sampler=sampler.__name__, beta=beta):
            np.random.seed(1729)
            legacy = sampler(*args, **keywords)
            after = np.random.get_state()
            reference = np.random.RandomState(1729)
            for shape in draw_shapes:
                reference.randn(*shape)
            assert_state_equal(after, reference.get_state())
            np.random.seed(1729)
            np.testing.assert_array_equal(sampler(*args, **keywords, rng=None), legacy)
            assert_state_equal(np.random.get_state(), after)
            first, second = np.random.default_rng(91), np.random.default_rng(91)
            with monkeypatch.context() as patch:

                def forbidden(*shape: Any) -> Any:
                    raise AssertionError("Explicit Generator accessed the global RNG")

                patch.setattr(np.random, "randn", forbidden)
                actual = sampler(*args, **keywords, rng=first)
                assert actual.dtype == legacy.dtype
                expected_stream = np.random.default_rng(91)
                for shape in draw_shapes:
                    expected_stream.standard_normal(shape)
                assert first.bit_generator.state == expected_stream.bit_generator.state
                np.testing.assert_array_equal(
                    actual, sampler(*args, **keywords, rng=second)
                )
                assert not np.array_equal(actual, sampler(*args, **keywords, rng=first))
                for invalid in (7, np.random.RandomState(91), object()):
                    with pytest.raises(TypeError, match="Generator or None"):
                        sampler(*args, **keywords, rng=invalid)
            assert_state_equal(np.random.get_state(), after)
            if "haar" in sampler.__name__:
                np.testing.assert_allclose(
                    actual @ actual.conj().T, np.eye(len(actual)), atol=2e-14
                )
                if beta == 4:
                    j = symplectic_form(3)
                    np.testing.assert_allclose(actual.T @ j @ actual, j, atol=2e-14)
            else:
                assert_hermitian_structure(actual, 3, beta)
                if sampler is sample_wishart:
                    assert np.linalg.eigvalsh(actual).min() >= -1e-12
