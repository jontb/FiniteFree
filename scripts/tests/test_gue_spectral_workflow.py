"""End-to-end composition without statistical pass/fail assertions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "examples" / "gue_spectral_workflow.py"
spec = importlib.util.spec_from_file_location("gue_spectral_workflow", MODULE)
assert spec is not None and spec.loader is not None
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_workflow_runs_and_retains_sampling_units() -> None:
    state = np.random.get_state()
    result = workflow.run_workflow(dimension=3, samples=8, seed=81)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    eigs = np.asarray(result["sampled_eigenvalues"])
    assert eigs.shape == (8, 3)
    np.testing.assert_allclose(
        result["mean_esd_moments_mass_mean_second"], [1, 0, 1], atol=1e-14
    )
    assert result["expected_polynomial_root_second_moment"] == pytest.approx(2 / 3)
    assert result["sample_mean_second_moment"] == pytest.approx(np.mean(eigs**2))
    assert result["sample_second_moment_standard_error"] == pytest.approx(
        np.std(np.mean(eigs**2, axis=1), ddof=1) / np.sqrt(8)
    )
    assert result["gap_quadrature_change"] < 1e-12
    assert 0 <= result["gap_nystrom_80"] <= 1
    assert isinstance(result["coefficient_confidence_check"], bool)
    assert len(result["expected_characteristic_coefficients"]) == 4
    lower, upper = result["sample_gap_wilson_95_interval"]
    assert 0 <= lower <= result["sample_gap_frequency"] <= upper <= 1


def test_invalid_workflow_sizes() -> None:
    for dimension, samples in [(0, 2), (2, 1)]:
        with pytest.raises(ValueError):
            workflow.run_workflow(dimension, samples)
