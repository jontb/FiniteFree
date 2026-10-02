import math
from typing import Any

import numpy as np
import pytest

from finitefree import EmpiricalComparison, RealRootedPolynomial


def comparison_from_scalars(target: float, values: list[float]) -> EmpiricalComparison:
    matrices = iter([np.array([[value]]) for value in values])
    return EmpiricalComparison(
        RealRootedPolynomial.from_roots([target]), len(values), lambda: next(matrices)
    )


def test_deterministic_mismatch_is_not_skipped() -> None:
    p = RealRootedPolynomial.from_roots([1, 2])
    comparison = EmpiricalComparison(p, 3, lambda: np.zeros((2, 2)))
    assert not comparison.verify_coefficients()


@pytest.mark.parametrize("scale", [1e-250, 1.0, 1e250])
def test_deterministic_and_low_variance_comparisons_preserve_scale(
    scale: float,
) -> None:
    assert comparison_from_scalars(scale, [scale] * 3).verify_coefficients()
    assert not comparison_from_scalars(0, [scale] * 3).verify_coefficients()
    assert not comparison_from_scalars(
        0, [scale * (1 + 1e-6 * k) for k in [-1, 0, 1]]
    ).verify_coefficients()


@pytest.mark.parametrize("scale", [1e-250, 1.0, 1e250])
@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_t_band_matches_closed_form_two_degree_reference(
    scale: float, alpha: float
) -> None:
    # For samples [-1, 0, 1], SEM=1/sqrt(3), df=2. Integrating the
    # t_2 density gives sf(t)=(1-t/sqrt(t*t+2))/2, so its inverse is
    # (1-2u)/sqrt(2u(1-u)). Two polynomial coefficients share alpha.
    u = alpha / 4
    critical = (1 - 2 * u) / math.sqrt(2 * u * (1 - u))
    width = critical / math.sqrt(3)
    values = [-scale, 0, scale]
    assert comparison_from_scalars(0.99 * width * scale, values).verify_coefficients(
        alpha=alpha, rtol=0
    )
    assert not comparison_from_scalars(
        1.01 * width * scale, values
    ).verify_coefficients(alpha=alpha, rtol=0)


def test_alpha_and_roundoff_tolerances_affect_comparison() -> None:
    comparison = comparison_from_scalars(2, [-1, 0, 1])
    assert comparison.verify_coefficients(alpha=0.05)
    assert not comparison.verify_coefficients(alpha=0.2)
    deterministic = comparison_from_scalars(0, [1e-30] * 3)
    assert not deterministic.verify_coefficients()
    assert deterministic.verify_coefficients(atol=1e-30)
    nearby = comparison_from_scalars(1, [1 + 1e-11] * 3)
    assert nearby.verify_coefficients()
    assert not nearby.verify_coefficients(rtol=0)


@pytest.mark.parametrize("samples", [0, 1, -1, 2.5, True])
def test_sample_count_rejected_before_generator_runs(samples: Any) -> None:
    def unexpected_generator() -> Any:
        raise AssertionError("invalid sample count reached generator")

    with pytest.raises(ValueError, match="integer of at least 2"):
        EmpiricalComparison(
            RealRootedPolynomial.from_roots([1]), samples, unexpected_generator
        )


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.eye(3), "shape"),
        (np.zeros((2, 3)), "shape"),
        (np.array([[0, 1], [0, 0]]), "Hermitian"),
        (np.eye(2) * 1j, "Hermitian"),
        (np.full((2, 2), np.nan), "finite entries"),
        (np.full((2, 2), np.inf), "finite entries"),
        (np.diag([1, 2, 3, 4]), "paired eigenvalues"),
        (np.diag([1, 2, 3, 4]) * 1e-250, "paired eigenvalues"),
        (np.eye(2) * 1e200, "coefficients must be finite"),
    ],
)
def test_invalid_samples_are_rejected(matrix: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmpiricalComparison(RealRootedPolynomial.from_roots([1, 2]), 3, lambda: matrix)


@pytest.mark.parametrize("doubled", [False, True])
def test_paired_eigenvalues_and_complex_hermitian_samples(doubled: bool) -> None:
    # This Hermitian matrix has roots 1 and 3. Its doubled block representation
    # has two copies of each root, with characteristic square root x^2-4x+3.
    matrix = np.array([[2, 1j], [-1j, 2]])
    p = RealRootedPolynomial.from_roots([1, 3])
    sample = np.kron(np.eye(2), matrix) if doubled else matrix
    comparison = EmpiricalComparison(p, np.int64(3), lambda: sample)
    np.testing.assert_allclose(comparison.char_poly_coeffs, [[1, -4, 3]] * 3)
    assert comparison.verify_coefficients()


def test_degree_zero_comparison_has_one_coefficient() -> None:
    comparison = EmpiricalComparison(
        RealRootedPolynomial([1]), 3, lambda: np.empty((0, 0))
    )
    assert comparison.char_poly_coeffs.shape == (3, 1)
    assert comparison.verify_coefficients()


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan, np.inf, 5e-324])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    comparison = comparison_from_scalars(1, [1] * 3)
    with pytest.raises(ValueError, match="alpha"):
        comparison.verify_coefficients(alpha=alpha)


@pytest.mark.parametrize("tolerances", [{"rtol": -1}, {"atol": -1}, {"rtol": np.inf}])
def test_invalid_tolerances_are_rejected(tolerances: dict[str, float]) -> None:
    comparison = comparison_from_scalars(1, [1] * 3)
    with pytest.raises(ValueError, match="rtol and atol"):
        comparison.verify_coefficients(**tolerances)


def test_nonfinite_analytical_coefficients_are_rejected() -> None:
    p = RealRootedPolynomial.from_roots([10**200, 10**200])
    comparison = EmpiricalComparison(p, 3, lambda: np.eye(2))
    with pytest.raises(ValueError, match="Analytical coefficients"):
        comparison.verify_coefficients()
