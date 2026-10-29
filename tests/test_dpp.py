import itertools
import math
import warnings
from typing import Any

import flint
import numpy as np
import pytest
import sympy as sp

from finitefree import (
    DiscreteFiniteKernel,
    PreparedDiscreteDPP,
    count_distribution_discrete,
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
            # Retain the three-state example with a deterministic occupied state.
            third = flint.fmpq(1, 3)
            kernel = DiscreteFiniteKernel(
                representation(
                    [[2 * third, third, 0], [third, 2 * third, 0], [0, 0, 1]]
                )
            )
            assert kernel(0, 0) == kernel.k_point_correlation([0]) == 2 * third
            assert kernel(0, 1) == third
            assert kernel(2, 2) == 1
            assert kernel.k_point_correlation([0, 1]) == third
            assert gap_probability_discrete(kernel, [0, 1]) == 0


def test_sampling_kernel_on_requested_states() -> None:
    kernel = DiscreteFiniteKernel(np.diag([0.0, 1.0, 1.0]))
    for states in ([2, 0], [2, 0, 1]):
        with case(states=states):
            assert set(sample_discrete(kernel, states)) == set(states) & {1, 2}
    assert sample_discrete(kernel, []) == []


def test_invalid_kernels_are_rejected_before_random_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid kernel consumed randomness")

    monkeypatch.setattr(np.random, "rand", unexpected)
    monkeypatch.setattr(np.random, "choice", unexpected)
    matrix: Any
    for prepared in (False, True):
        for matrix, states, message in [
            (np.eye(2) * 2, [0, 1], "eigenvalues"),
            (np.diag([-0.01, 0.5]), [0, 1], "eigenvalues"),
            (np.diag([-0.1, 0.5]), [0, 1], "eigenvalues"),
            (np.array([[0.5, 0.1], [0.0, 0.5]]), [0, 1], "symmetric"),
            (np.diag([np.nan, 0.5]), [0, 1], "finite"),
            (np.diag([np.inf, 0.5]), [0, 1], "finite"),
            (np.eye(3), [0, 1], "shape"),
            (np.array([0.5, 0.5]), [0, 1], "shape"),
            (np.eye(2, dtype=complex), [0, 1], "real symmetric"),
            (np.eye(2), [0, 0], "distinct"),
        ]:
            with case(prepared=prepared, matrix=matrix, states=states):
                with pytest.raises(ValueError, match=message):
                    if prepared:
                        PreparedDiscreteDPP(matrix, states)
                    else:
                        sample_discrete(matrix, states)


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


def test_complex_kernels_are_rejected_before_casting_or_random_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid kernel consumed randomness")

    monkeypatch.setattr(np.random, "rand", unexpected)
    monkeypatch.setattr(np.random, "choice", unexpected)
    for prepared, representation, entry in itertools.product(
        (False, True),
        ("array", "object", "callable"),
        (complex(0.5), complex(0.5, 0.25), np.complex64(0.5), np.complex128(0.5)),
    ):
        with case(prepared=prepared, representation=representation, entry=entry):
            matrix = np.array(
                [[entry]], dtype=object if representation == "object" else None
            )
            kernel = (
                DiscreteFiniteKernel(matrix) if representation == "callable" else matrix
            )
            # A lossy cast must not be accepted merely because it emits a warning.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
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


def _discrete_representations() -> list[tuple[str, Any]]:
    return [
        ("list", lambda rows: [list(row) for row in rows]),
        ("ndarray", np.array),
        ("fmpq_mat", flint.fmpq_mat),
    ]


def test_discrete_gap_requires_distinct_states() -> None:
    q = flint.fmpq
    rows = [[q(1, 2), q(1, 4), 0], [q(1, 4), q(1, 2), 0], [0, 0, q(1, 3)]]
    # Hand-computed det(I - K_I): 1 - K_ii on singletons, (1/2)^2 - (1/4)^2 on
    # the dense block and a factor 1 - 1/3 = 2/3 for the independent state.
    expected = {
        (0,): q(1, 2),
        (1,): q(1, 2),
        (2,): q(2, 3),
        (0, 1): q(3, 16),
        (2, 0): q(1, 3),
        (0, 1, 2): q(1, 8),
    }
    for name, representation in _discrete_representations():
        kernel = DiscreteFiniteKernel(representation(rows))
        for points, value in expected.items():
            with case(representation=name, points=points):
                assert gap_probability_discrete(kernel, points) == value
        for repeated in ([0, 0], [2, 0, 2], [np.int64(1), 1]):
            with case(representation=name, repeated=repeated):
                # A repeated state would give det(I - K) of a singular duplicate.
                with pytest.raises(ValueError, match="distinct"):
                    gap_probability_discrete(kernel, repeated)


def test_discrete_kernel_rejects_negative_indices_for_every_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid state index consumed randomness")

    diagonal = [0.2, 0.3, 0.4, 0.5, 0.4468]
    q = flint.fmpq
    exact_diagonal = [q(2, 10), q(3, 10), q(4, 10), q(5, 10), q(4468, 10000)]
    for name, representation in _discrete_representations():
        values = exact_diagonal if name == "fmpq_mat" else diagonal
        rows = [[values[i] if i == j else 0 for j in range(5)] for i in range(5)]
        kernel = DiscreteFiniteKernel(representation(rows))
        for i, j in [(-1, 0), (0, -1), (-5, -5), (np.int64(-1), 4), (5, 0), (0, 5)]:
            with case(representation=name, index=(i, j)):
                with pytest.raises(IndexError):
                    kernel(i, j)
        for points in ([4, -1], [-1], [0, -5]):
            with case(representation=name, points=points):
                # -1 must not alias state 4 and yield its duplicated law.
                with pytest.raises(IndexError):
                    count_distribution_discrete(kernel, points)
                with pytest.raises(IndexError):
                    gap_probability_discrete(kernel, points)
                with monkeypatch.context() as patch:
                    patch.setattr(np.random, "rand", unexpected)
                    patch.setattr(np.random, "choice", unexpected)
                    with pytest.raises(IndexError):
                        sample_discrete(kernel, points)
        with case(representation=name, points=[4, 3]):
            # Independent Bernoulli enumeration for the diagonal restriction.
            p = [diagonal[4], diagonal[3]]
            expected = np.zeros(3)
            for occupied in itertools.product((0, 1), repeat=2):
                expected[sum(occupied)] += math.prod(
                    p[k] if bit else 1 - p[k] for k, bit in enumerate(occupied)
                )
            np.testing.assert_allclose(
                count_distribution_discrete(kernel, [4, 3]).pmf(),
                expected,
                rtol=0,
                atol=1e-15,
            )
            assert gap_probability_discrete(kernel, [4]) == 1 - values[4]


def test_discrete_kernel_owns_a_snapshot_of_its_matrix() -> None:
    q = flint.fmpq
    rows = [[q(1, 2), q(1, 4)], [q(1, 4), q(1, 2)]]
    float_rows = [[0.5, 0.25], [0.25, 0.5]]
    sources: list[tuple[str, Any]] = [
        ("list", [list(row) for row in float_rows]),
        ("ndarray", np.array(float_rows)),
        ("object ndarray", np.array(rows, dtype=object)),
        ("fmpq_mat", flint.fmpq_mat(rows)),
    ]
    for name, source in sources:
        with case(representation=name):
            entry = source[0, 0] if name == "fmpq_mat" else source[0][0]
            off_diagonal = source[0, 1] if name == "fmpq_mat" else source[0][1]
            kernel = DiscreteFiniteKernel(source)
            if name == "fmpq_mat":
                source[0, 0] = q(9, 10)
                source[0, 1] = q(0)
            else:
                source[0][0] = 0.9
                source[0][1] = 0
            # Entry types are unchanged and caller edits do not reach the kernel.
            assert type(kernel(0, 0)) is type(entry)
            assert kernel(0, 0) == entry and kernel(0, 1) == off_diagonal
            # 1/2 * 1/2 - 1/4 * 1/4 is exact in binary floating point too.
            assert float(gap_probability_discrete(kernel, [0, 1])) == 3 / 16
