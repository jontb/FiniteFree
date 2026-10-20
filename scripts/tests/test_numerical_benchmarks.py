"""Benchmark receipts must reject wrong answers before admitting timings."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from finitefree.hyperbolic import PencilFactorization
from finitefree.numerical import PreparedOrthogonalKernel
from finitefree.streaming import StreamingEmpiricalComparison
from scripts import run_numerical_benchmarks as runner
from scripts.benchmark_numerical_workflows import benchmark_kernel, benchmark_summary
from scripts.benchmark_prepared_pencils import benchmark, independent_reference
from scripts.run_numerical_benchmarks import check_pairs, validate_receipt
from tests._cases import case


def test_pencil_reference_matches_closed_form_diagonal_derivatives() -> None:
    a = np.array([np.eye(2), np.diag([2.0, -3.0])])
    x = np.array([[4.0, 0.5]])
    v = np.array([0.25, -0.5])
    sign, log, gradient, hvp = independent_reference(a, x, v)
    diagonal = np.array([5.0, 2.5])
    slopes = np.array([[1.0, 1.0], [2.0, -3.0]])
    np.testing.assert_allclose(sign, [1])
    np.testing.assert_allclose(log, [np.log(12.5)])
    np.testing.assert_allclose(gradient, [np.sum(slopes / diagonal, axis=1)])
    np.testing.assert_allclose(
        hvp, [-np.sum(slopes * (v @ slopes) / diagonal**2, axis=1)]
    )


def test_pencil_gate_catches_a_fault_shared_with_its_cpu_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert benchmark("numpy", 2, 2, [1], 1)["status"] == "measured"
    original = PencilFactorization.logabsdet_hvp

    def wrong(self: Any, direction: Any) -> Any:
        return original(self, direction) + 0.25

    monkeypatch.setattr(PencilFactorization, "logabsdet_hvp", wrong)
    result = benchmark("numpy", 2, 2, [1], 1)
    assert (
        result["records"][0]["cpu_reference_scaled_errors_sign_log_gradient_hvp"]
        == [0] * 4
    )
    assert result["status"] == "invalid"


def test_case_input_does_not_depend_on_preceding_batches() -> None:
    combined = benchmark("numpy", 2, 2, [1, 3], 1)
    alone = benchmark("numpy", 2, 2, [3], 1)
    assert combined["status"] == alone["status"] == "measured"
    assert combined["records"][1]["input_sha256"] == alone["records"][0]["input_sha256"]


def test_kernel_gate_rejects_wrong_prepared_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert benchmark_kernel("prepared", 3, 3, 1)["status"] == "measured"
    original = PreparedOrthogonalKernel.matrix

    def wrong(self: Any, xs: Any, ys: Any = None) -> Any:
        return original(self, xs, ys) + 0.25

    monkeypatch.setattr(PreparedOrthogonalKernel, "matrix", wrong)
    assert benchmark_kernel("prepared", 3, 3, 1)["status"] == "invalid"


def test_summary_gate_rejects_wrong_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    assert benchmark_summary("streaming", 2, 4, 1)["status"] == "measured"
    original = vars(StreamingEmpiricalComparison)["count_histograms"].fget
    monkeypatch.setattr(
        StreamingEmpiricalComparison,
        "count_histograms",
        property(lambda self: original(self) + 1),
    )
    assert benchmark_summary("streaming", 2, 4, 1)["status"] == "invalid"


def test_receipts_require_matched_inputs_sources_and_true_accuracy_gates() -> None:
    check_pairs(
        {
            "kernel-4-4-scalar": {"input_sha256": "same"},
            "kernel-4-4-prepared": {"input_sha256": "same"},
        }
    )
    with pytest.raises(ValueError, match="identical inputs"):
        check_pairs(
            {
                "kernel-4-4-scalar": {"input_sha256": "first"},
                "kernel-4-4-prepared": {"input_sha256": "second"},
            }
        )
    receipt = {
        "status": "measured",
        "source_and_tool_sha256": "baseline",
        "accuracy_passed": True,
    }
    with pytest.raises(ValueError, match="snapshot changed"):
        validate_receipt(receipt, "candidate")
    validate_receipt(receipt, "baseline")
    overrides: dict[str, Any]
    for overrides in (
        {"accuracy_passed": False},
        {"records": []},
        {"records": [{"accuracy_passed": False}]},
        {"status": "incomplete"},
    ):
        with case(overrides=overrides), pytest.raises(ValueError):
            validate_receipt(receipt | overrides, "baseline")


def test_timeout_preserves_completed_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "receipt"
    monkeypatch.setattr(sys, "argv", ["runner", "--output", str(output)])
    monkeypatch.setattr(
        runner, "metadata", lambda script: {"source_and_tool_sha256": "fixed"}
    )
    monkeypatch.setattr(
        runner,
        "build_cases",
        lambda profile, gpu: [
            ("kernel-2-2-scalar", ["first"]),
            ("kernel-2-2-prepared", ["second"]),
        ],
    )

    def child(command: list[str], **kwargs: Any) -> None:
        if "second" in command:
            raise subprocess.TimeoutExpired(command, 1)
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "status": "measured",
                    "accuracy_passed": True,
                    "source_and_tool_sha256": "fixed",
                    "input_sha256": "same",
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", child)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["completed"] == ["kernel-2-2-scalar"]
    assert "TimeoutExpired" in manifest["error"]
    assert (output / "kernel-2-2-scalar.json").exists()
    assert (output / "kernel-2-2-prepared.log").exists()
