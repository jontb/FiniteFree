"""Deterministic Gram/count identities plus a small, non-statistical smoke run."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from scipy.special import erf

from finitefree import gue_expected_poly

MODULE = Path(__file__).resolve().parents[2] / "examples" / "gue_concentration.py"
spec = importlib.util.spec_from_file_location("gue_concentration", MODULE)
assert spec is not None and spec.loader is not None
showcase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(showcase)


@pytest.mark.parametrize("n", [1, 2, 6])
def test_basis_matches_exact_projected_polynomials(n: int) -> None:
    x = np.array([-0.5, 0.0, 0.25])
    values = showcase.normalized_basis(n, x)
    p = gue_expected_poly(n)
    for j in range(n):
        projected = p.projection(j)
        norm = math.factorial(j) / n**j
        reference = [
            float(projected.evaluate(sp.Rational(t))) / math.sqrt(norm) for t in x
        ]
        np.testing.assert_allclose(values[:, j], reference, atol=2e-15)


@pytest.mark.parametrize("n", [1, 2])
def test_gram_against_gaussian_integrals(n: int) -> None:
    a = 0.5
    gram = showcase.restricted_gram(n, (-a, a), 48)
    mass = erf(a * math.sqrt(n / 2))
    assert gram[0, 0] == pytest.approx(mass, abs=2e-15)
    if n == 2:
        # Integration by parts: integral n*x^2*w_n = mass-2*a*w_n(a).
        second = mass - 2 * a * math.sqrt(n / (2 * math.pi)) * math.exp(-n * a * a / 2)
        assert gram[1, 1] == pytest.approx(second, abs=2e-15)
        assert abs(gram[0, 1]) < 1e-16
    law = showcase.count_law(gram)
    pmf = np.asarray(law["pmf"])
    k = np.arange(n + 1)
    assert np.sum(pmf) == pytest.approx(1.0)
    assert k @ pmf == pytest.approx(law["trace_mean"])
    assert (k - law["mean_count"]) ** 2 @ pmf == pytest.approx(law["trace_variance"])
    assert pmf[0] == pytest.approx(np.linalg.det(np.eye(n) - gram))
    if n == 1:
        np.testing.assert_allclose(pmf, [1 - mass, mass])


def test_count_law_bound_and_refinement() -> None:
    n = 8
    theory = showcase.theory_case(n, (-0.5, 0.5), 32)
    assert theory["gram_refinement_max_abs"] < 1e-12
    assert theory["pmf_refinement_l1"] < 1e-12
    assert theory["kernel_trace_disagreement_abs"] < 1e-12
    assert theory["gap_disagreement_abs"] < 1e-12
    pmf = np.asarray(theory["pmf"])
    for epsilon in [0.0, 0.01, 0.05, 0.1, 0.3]:
        tail = np.sum(
            pmf[np.abs(np.arange(n + 1) / n - theory["mean_fraction"]) >= epsilon]
        )
        assert tail <= showcase.bernstein(n, theory["variance_count"], epsilon) + 1e-14


def test_invalid_gram_is_not_silently_clipped() -> None:
    for matrix in [
        np.diag([-0.1, 0.5]),
        np.diag([0.5, 1.1]),
        np.array([[1.0, 0.1], [0.2, 1.0]]),
    ]:
        with pytest.raises(ValueError):
            showcase.count_law(matrix)
    law = showcase.count_law(np.diag([-1e-16, 0.5]))
    assert law["roundoff_clip_max"] == 1e-16


def test_interval_counts_use_one_matrix_per_row() -> None:
    eigs = np.array([[-0.5, 0.0, 0.5], [-1.0, 0.25, 2.0]])
    counts = showcase.interval_counts(eigs, (-0.5, 0.5))
    np.testing.assert_array_equal(counts, [3, 1])
    np.testing.assert_allclose(counts / 3, [1.0, 1 / 3])


def test_binomial_limits_preserve_unresolved_zero_tail() -> None:
    low, high = showcase.binomial_interval(0, 2000)
    assert low == 0
    assert high == pytest.approx(1 - 0.025 ** (1 / 2000))
    assert high > 1 - 0.05 ** (1 / 2000) > 0
    low, high = showcase.binomial_interval(2000, 2000)
    assert high == 1 and low == pytest.approx(0.025 ** (1 / 2000))
    low, high = showcase.binomial_interval(5, 10)
    assert low == pytest.approx(1 - high)
    assert showcase.bernstein(16, 0.5, 0.0) == 1.0


def test_bounded_reproducible_smoke() -> None:
    state = np.random.get_state()
    report = showcase.run_showcase([2], samples=8, seed=53, order=16)
    after = np.random.get_state()
    assert isinstance(state, tuple) and isinstance(after, tuple)
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    row = report["records"][0]
    np.testing.assert_array_equal(
        row["observed_counts"], showcase.sample_counts(2, 8, 55, (-0.5, 0.5))
    )
    assert len(row["observed_counts"]) == 8
    assert row["tails"][0]["numerical_count_law_tail"] == 1
    assert not row["tails"][0]["theory_outside_pointwise_interval"]
    for tail in row["tails"]:
        assert 0 <= tail["numerical_count_law_tail"] <= 1
        assert len(tail["pointwise_cp95"]) == 2
        if tail["hits"] == 0:
            assert tail["zero_hit_one_sided_95_upper"] > 0
    # Do not assert that a statistical confidence interval happens to cover.
    with pytest.raises(ValueError, match="2000"):
        showcase.run_showcase([2], samples=2001, seed=1)
