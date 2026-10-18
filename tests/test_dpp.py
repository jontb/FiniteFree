import itertools
import math
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    DiscreteFiniteKernel,
    PreparedDiscreteDPP,
    gap_probability_discrete,
    sample_discrete,
)
from tests._cases import case


def test_exact_kernel_representations() -> None:
    representation: Any
    for representation in [list, np.array, flint.fmpq_mat]:
        with case(representation=representation):
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


def test_invalid_sampling_kernels() -> None:
    matrix: Any
    message: str
    for matrix, message in [
        (np.eye(2) * 2, "eigenvalues"),
        (np.diag([-0.01, 0.5]), "eigenvalues"),
        (np.array([[0.5, 0.1], [0.0, 0.5]]), "symmetric"),
        (np.diag([np.nan, 0.5]), "finite"),
        (np.diag([np.inf, 0.5]), "finite"),
        (np.eye(3), "shape"),
        (np.array([0.5, 0.5]), "shape"),
        (np.eye(2, dtype=complex), "real symmetric"),
    ]:
        with case(matrix=matrix, message=message):
            with pytest.raises(ValueError, match=message):
                sample_discrete(matrix, [0, 1])


def test_sampling_boundaries_preserve_state_labels_and_projection_rank() -> None:
    matrix = np.diag([-1e-12, 1 + 1e-12])
    states = ["excluded", "included"]
    assert sample_discrete(matrix, states) == ["included"]
    assert PreparedDiscreteDPP(matrix, states).sample() == ["included"]
    assert PreparedDiscreteDPP(np.empty((0, 0)), []).sample() == []
    with pytest.raises(ValueError, match="distinct"):
        sample_discrete(np.eye(2), [0, 0])
    with pytest.raises(ValueError, match="shape"):
        PreparedDiscreteDPP(np.eye(1), [])
    with pytest.raises(ValueError, match="required"):
        sample_discrete(np.eye(1))
    with pytest.raises(TypeError, match="Generator"):
        PreparedDiscreteDPP(matrix, states).sample(rng=123)  # type: ignore[arg-type]
    kernel = DiscreteFiniteKernel(np.diag([1.0, 1.0, 1.0, 0.0, 0.0]))
    for _ in range(5):
        sample = sample_discrete(kernel, list(range(5)))
        assert len(sample) == 3
        assert set(sample) == {0, 1, 2}


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("representation", ["array", "object", "callable"])
@pytest.mark.parametrize("entry", [complex(0.5), complex(0.5, 0.25)])
def test_complex_kernels_are_rejected_before_casting_or_random_draws(
    prepared: bool,
    representation: str,
    entry: complex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.array([[entry]], dtype=object if representation == "object" else None)
    kernel = DiscreteFiniteKernel(matrix) if representation == "callable" else matrix

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid kernel consumed randomness")

    monkeypatch.setattr(np.random, "rand", unexpected)
    with pytest.raises(ValueError, match="real symmetric"):
        if prepared:
            PreparedDiscreteDPP(kernel, [0])
        else:
            sample_discrete(kernel, [0])


def test_raw_and_prepared_sampling_against_exact_subset_laws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernels = [
        (
            "nonprojection",
            sp.Matrix(
                [
                    [sp.Rational(3, 5), sp.Rational(1, 5)],
                    [sp.Rational(1, 5), sp.Rational(2, 5)],
                ]
            ),
        ),
        ("projection", sp.eye(3) - sp.ones(3) / 3),
        ("dense", sp.Matrix([[5, 2, -1], [2, 6, 1], [-1, 1, 4]]) / 10),
    ]
    scenarios = [(name, exact, "prepared", 8000, 20261006) for name, exact in kernels]
    scenarios += [
        (kernels[0][0], kernels[0][1], "legacy", 4000, 12345),
        (kernels[1][0], kernels[1][1], "legacy", 3000, 20261002),
    ]
    for name, exact, mode, trials, seed in scenarios:
        with case(kernel=name, mode=mode):
            n = exact.rows
            subsets = [
                subset
                for size in range(n + 1)
                for subset in itertools.combinations(range(n), size)
            ]
            expected = np.zeros(1 << n)
            for subset in subsets:
                probability = sum(
                    (-1) ** (len(superset) - len(subset))
                    * exact.extract(superset, superset).det()
                    for superset in subsets
                    if set(subset) <= set(superset)
                )
                assert probability >= 0
                expected[sum(1 << i for i in subset)] = float(probability)
            np.testing.assert_allclose(expected.sum(), 1, rtol=0, atol=2e-16)
            matrix = np.array(exact, dtype=float)
            if mode == "legacy" and name == "projection":
                matrix = np.eye(3) - np.ones((3, 3)) / 3
            states = list(range(n)) if mode == "legacy" else [101, -7, 43][:n]
            prepared = PreparedDiscreteDPP(matrix, states)
            rng = np.random.default_rng(seed)
            counts = np.zeros(len(expected))
            with monkeypatch.context() as patch:
                patch.setattr(np.random, "rand", rng.random)
                patch.setattr(np.random, "choice", rng.choice)
                for _ in range(trials):
                    sample = (
                        sample_discrete(matrix, states)
                        if mode == "legacy"
                        else prepared.sample(rng=rng)
                    )
                    assert len(sample) == len(set(sample))
                    assert set(sample) <= set(states)
                    if name == "projection":
                        assert len(sample) == 2
                    counts[sum(1 << states.index(state) for state in sample)] += 1
            # Preserve the legacy bounds and the simultaneous Hoeffding bound.
            tolerance = (
                0.03
                if mode == "legacy"
                else math.sqrt(math.log(2 * len(expected) / 1e-5) / (2 * trials))
            )
            np.testing.assert_allclose(
                counts / trials, expected, rtol=0, atol=tolerance
            )
