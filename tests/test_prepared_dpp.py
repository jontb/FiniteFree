from typing import Any

import numpy as np
import pytest

from finitefree import DiscreteFiniteKernel, PreparedDiscreteDPP, sample_discrete
from tests._cases import case


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


def test_prepared_snapshots_own_matrices_labels_and_exports() -> None:
    for callable_source in (False, True):
        with case(callable_source=callable_source):
            matrix = np.diag([0.0, 1.0, 1.0] if callable_source else [0.0, 1.0])
            states: Any = [2, 0] if callable_source else ["excluded", "included"]
            kernel = DiscreteFiniteKernel(matrix) if callable_source else matrix
            prepared = PreparedDiscreteDPP(kernel, states)
            expected_states = tuple(states)
            expected_matrix = np.diag([1.0, 0.0] if callable_source else [0.0, 1.0])
            matrix[:] = 0
            states.reverse()
            exported, values = prepared.kernel_matrix, prepared.eigenvalues
            assert not exported.flags.writeable and not values.flags.writeable
            exported.setflags(write=True)
            values.setflags(write=True)
            exported[:] = 0
            values[:] = 0
            assert prepared.state_space == expected_states
            assert prepared.sample() == ([2] if callable_source else ["included"])
            np.testing.assert_array_equal(prepared.kernel_matrix, expected_matrix)
            with pytest.raises(ValueError, match="state order"):
                sample_discrete(prepared, states)
            with pytest.raises(AttributeError):
                prepared.state_space = ("changed",)  # type: ignore[misc]


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


def test_sampler_representations_preserve_preparation_and_rng_sequences() -> None:
    state = np.random.get_state()
    try:
        matrix = np.array([[0.6, 0.2], [0.2, 0.4]])
        for representation, labels in (
            ("array", ["a", "b"]),
            ("array", [0, 1]),
            ("object", [0, 1]),
            ("callable", [0, 1]),
        ):
            with case(representation=representation, labels=labels):
                kernel: Any = (
                    matrix.astype(object) if representation == "object" else matrix
                )
                if representation == "callable":
                    kernel = DiscreteFiniteKernel(kernel)
                np.random.seed(20261006)
                before: Any = np.random.get_state()
                prepared = PreparedDiscreteDPP(kernel, labels)
                after: Any = np.random.get_state()
                assert before[0] == after[0] and before[2:] == after[2:]
                np.testing.assert_array_equal(before[1], after[1])
                for seed in range(20):
                    expected = sample_discrete(
                        kernel, labels, rng=np.random.default_rng(seed)
                    )
                    assert prepared.sample(rng=np.random.default_rng(seed)) == expected
                    assert (
                        sample_discrete(prepared, rng=np.random.default_rng(seed))
                        == expected
                    )
                    assert (
                        sample_discrete(
                            prepared, labels, rng=np.random.default_rng(seed)
                        )
                        == expected
                    )
                    unchanged: Any = np.random.get_state()
                    assert after[0] == unchanged[0] and after[2:] == unchanged[2:]
                    np.testing.assert_array_equal(after[1], unchanged[1])
                    np.random.seed(seed)
                    expected = sample_discrete(kernel, labels)
                    np.random.seed(seed)
                    assert prepared.sample() == expected
                    after = np.random.get_state()
    finally:
        np.random.set_state(state)
