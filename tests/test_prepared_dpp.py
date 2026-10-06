from typing import Any

import numpy as np
import pytest

from finitefree import DiscreteFiniteKernel, PreparedDiscreteDPP, sample_discrete


@pytest.mark.parametrize("seed", range(20))
def test_prepared_and_raw_seeded_samples_agree(seed: int) -> None:
    kernel = np.array([[0.6, 0.2], [0.2, 0.4]])
    prepared = PreparedDiscreteDPP(kernel, ["a", "b"])
    expected = sample_discrete(kernel, ["a", "b"], rng=np.random.default_rng(seed))
    assert prepared.sample(rng=np.random.default_rng(seed)) == expected
    assert sample_discrete(prepared, rng=np.random.default_rng(seed)) == expected
    assert (
        sample_discrete(prepared, ["a", "b"], rng=np.random.default_rng(seed))
        == expected
    )


def test_prepared_does_not_redecompose_or_revalidate_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = np.linalg.eigh

    def counted(matrix: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    prepared = PreparedDiscreteDPP(np.eye(3) * 0.5, [0, 1, 2])
    assert calls == 1
    for _ in range(10):
        sample_discrete(prepared)
    assert calls == 1


def test_prepared_owns_original_inputs_and_exports() -> None:
    matrix = np.diag([0.0, 1.0])
    states = ["excluded", "included"]
    prepared = PreparedDiscreteDPP(matrix, states)
    matrix[:] = 0
    states.reverse()
    matrix_export = prepared.kernel_matrix
    values_export = prepared.eigenvalues
    assert not matrix_export.flags.writeable and not values_export.flags.writeable
    matrix_export.setflags(write=True)
    values_export.setflags(write=True)
    matrix_export[:] = 0
    values_export[:] = 0
    assert prepared.state_space == ("excluded", "included")
    assert prepared.sample() == ["included"]
    np.testing.assert_array_equal(prepared.kernel_matrix, np.diag([0.0, 1.0]))
    with pytest.raises(AttributeError):
        prepared.state_space = ("changed",)  # type: ignore[misc]


def test_callable_snapshot_respects_subsets_and_order() -> None:
    matrix = np.diag([0.0, 1.0, 1.0])
    source = DiscreteFiniteKernel(matrix)
    prepared = PreparedDiscreteDPP(source, [2, 0])
    matrix[:] = 0
    assert prepared.sample() == [2]
    assert prepared.state_space == (2, 0)
    with pytest.raises(ValueError, match="state order"):
        sample_discrete(prepared, [0, 2])


@pytest.mark.parametrize(
    "matrix,states",
    [
        (np.eye(2) * 2, [0, 1]),
        (np.diag([-0.1, 0.5]), [0, 1]),
        (np.eye(3), [0, 1]),
        (np.eye(2), [0, 0]),
        (np.diag([np.nan, 0.5]), [0, 1]),
        (np.array([[0.5, 0.1], [0, 0.5]]), [0, 1]),
        (np.eye(2, dtype=complex), [0, 1]),
    ],
)
def test_preparation_rejects_invalid_kernels_before_rng(
    matrix: Any, states: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid preparation drew random numbers")

    monkeypatch.setattr(np.random, "rand", unexpected)
    with pytest.raises(ValueError):
        PreparedDiscreteDPP(matrix, states)


def test_preparation_does_not_consume_rng_or_change_legacy_sequence() -> None:
    kernel = np.array([[0.6, 0.2], [0.2, 0.4]])
    np.random.seed(20261006)
    before: Any = np.random.get_state()
    prepared = PreparedDiscreteDPP(kernel, [0, 1])
    after: Any = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    for seed in range(20):
        np.random.seed(seed)
        expected = sample_discrete(kernel, [0, 1])
        np.random.seed(seed)
        assert prepared.sample() == expected


@pytest.mark.parametrize("projection", [False, True])
def test_prepared_exact_small_dpp_distribution(projection: bool) -> None:
    expected: Any
    if projection:
        kernel = np.eye(3) - np.ones((3, 3)) / 3
        expected = np.zeros(8)
        expected[[3, 5, 6]] = 1 / 3
    else:
        kernel = np.array([[0.6, 0.2], [0.2, 0.4]])
        expected = np.array([0.2, 0.4, 0.2, 0.2])
    prepared = PreparedDiscreteDPP(kernel, list(range(len(kernel))))
    rng = np.random.default_rng(20261006)
    counts = np.zeros(len(expected))
    trials = 4000
    for _ in range(trials):
        sample = prepared.sample(rng=rng)
        assert len(sample) == len(set(sample))
        counts[sum(1 << state for state in sample)] += 1
    # Principal minors and marginals give the independent exact laws above.
    np.testing.assert_allclose(counts / trials, expected, atol=0.035)


def test_empty_and_clipped_boundary_kernels() -> None:
    assert PreparedDiscreteDPP(np.empty((0, 0)), []).sample() == []
    prepared = PreparedDiscreteDPP(np.diag([-1e-12, 1 + 1e-12]), [0, 1])
    assert prepared.sample() == [1]
    with pytest.raises(ValueError, match="shape"):
        PreparedDiscreteDPP(np.eye(1), [])
    with pytest.raises(ValueError, match="required"):
        sample_discrete(np.eye(1))
    with pytest.raises(TypeError, match="Generator"):
        prepared.sample(rng=123)  # type: ignore[arg-type]
