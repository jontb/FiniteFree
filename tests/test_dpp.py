from typing import Any

import flint
import numpy as np
import pytest

from finitefree import DiscreteFiniteKernel, gap_probability_discrete, sample_discrete


@pytest.mark.parametrize("representation", [list, np.array, flint.fmpq_mat])
def test_exact_kernel_representations(representation: Any) -> None:
    matrix = [
        [flint.fmpq(1, 2), flint.fmpq(1, 4)],
        [flint.fmpq(1, 4), flint.fmpq(1, 2)],
    ]
    kernel = DiscreteFiniteKernel(representation(matrix))
    assert kernel(np.int64(0), 1) == flint.fmpq(1, 4)
    assert kernel.k_point_correlation([0, 1]) == flint.fmpq(3, 16)
    assert gap_probability_discrete(kernel, [0, 1]) == flint.fmpq(3, 16)
    with pytest.raises(TypeError):
        kernel(0.5, 1)


@pytest.mark.parametrize("states", [[2, 0], [2, 0, 1]])
def test_sampling_kernel_on_requested_states(states: list[int]) -> None:
    kernel = DiscreteFiniteKernel(np.diag([0.0, 1.0, 1.0]))
    assert set(sample_discrete(kernel, states)) == set(states) & {1, 2}
    assert sample_discrete(kernel, []) == []


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.eye(2) * 2, "eigenvalues"),
        (np.diag([-0.01, 0.5]), "eigenvalues"),
        (np.array([[0.5, 0.1], [0.0, 0.5]]), "symmetric"),
        (np.diag([np.nan, 0.5]), "finite"),
        (np.diag([np.inf, 0.5]), "finite"),
        (np.eye(3), "shape"),
        (np.array([0.5, 0.5]), "shape"),
        (np.eye(2, dtype=complex), "real symmetric"),
    ],
)
def test_invalid_sampling_kernels(matrix: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_discrete(matrix, [0, 1])


def test_sampling_roundoff_and_state_labels() -> None:
    matrix = np.diag([-1e-12, 1 + 1e-12])
    assert sample_discrete(matrix, ["excluded", "included"]) == ["included"]
    with pytest.raises(ValueError, match="distinct"):
        sample_discrete(np.eye(2), [0, 0])


def test_nondiagonal_projection_inclusion_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20261002)
    monkeypatch.setattr(np.random, "rand", rng.random)
    monkeypatch.setattr(np.random, "choice", rng.choice)
    # Projection onto the plane orthogonal to (1, 1, 1).
    kernel = np.eye(3) - np.ones((3, 3)) / 3
    pairs = np.zeros(3)
    trials = 3000
    for _ in range(trials):
        sample = sample_discrete(kernel, [0, 1, 2])
        assert len(sample) == len(set(sample)) == 2
        pairs[list({0, 1, 2} - set(sample))[0]] += 1
    # Each pair's inclusion probability is its 2x2 principal minor, 1/3.
    np.testing.assert_allclose(pairs / trials, np.full(3, 1 / 3), atol=0.03)


def test_nonprojection_sampling_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(12345)
    monkeypatch.setattr(np.random, "rand", rng.random)
    monkeypatch.setattr(np.random, "choice", rng.choice)
    kernel = np.array([[0.6, 0.2], [0.2, 0.4]])
    counts = np.zeros(4)
    trials = 4000
    for _ in range(trials):
        sample = sample_discrete(kernel, [0, 1])
        assert len(sample) == len(set(sample))
        counts[sum(1 << state for state in sample)] += 1
    # det(K)=.2, det(I-K)=.2, and marginals .6, .4 determine the law.
    np.testing.assert_allclose(counts / trials, [0.2, 0.4, 0.2, 0.2], atol=0.03)
