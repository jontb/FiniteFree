"""Run a fixed, bounded numerical benchmark plan in fresh child processes."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts._benchmark import ROOT, metadata, write_result


def build_cases(profile: str, gpu: bool) -> list[tuple[str, list[str]]]:
    repeats = "2" if profile == "smoke" else "5"
    pencils = (
        [(8, 2, 2)]
        if profile == "smoke"
        else list(itertools.product([16, 64, 128], [4], [1, 16, 128]))
    )
    kernels = (
        [(4, 4)]
        if profile == "smoke"
        else list(itertools.product([8, 32, 64], [16, 64]))
    )
    summaries = (
        [(4, 8)]
        if profile == "smoke"
        else list(itertools.product([8, 32], [128, 1024, 8192]))
    )
    cases = []
    for n, m, batch in pencils:
        for backend in ["numpy", "cupy"] if gpu else ["numpy"]:
            cases.append(
                (
                    f"pencil-{n}-{m}-{batch}-{backend}",
                    [
                        "scripts/benchmark_prepared_pencils.py",
                        "--backend",
                        backend,
                        "--size",
                        str(n),
                        "--variables",
                        str(m),
                        "--batches",
                        str(batch),
                        "--repeats",
                        repeats,
                    ],
                )
            )
    for rank, points in kernels:
        for method in ("scalar", "prepared"):
            cases.append(
                (
                    f"kernel-{rank}-{points}-{method}",
                    [
                        "scripts/benchmark_numerical_workflows.py",
                        "kernel",
                        "--method",
                        method,
                        "--rank",
                        str(rank),
                        "--points",
                        str(points),
                        "--repeats",
                        repeats,
                    ],
                )
            )
    for degree, samples in summaries:
        for method in ("retained", "streaming"):
            cases.append(
                (
                    f"summary-{degree}-{samples}-{method}",
                    [
                        "scripts/benchmark_numerical_workflows.py",
                        "summary",
                        "--method",
                        method,
                        "--degree",
                        str(degree),
                        "--samples",
                        str(samples),
                        "--repeats",
                        repeats,
                    ],
                )
            )
    return cases


def check_pairs(results: dict[str, Any]) -> None:
    groups: dict[str, list[str]] = {}
    for name, receipt in results.items():
        digest = (
            receipt["records"][0]["input_sha256"]
            if name.startswith("pencil-")
            else receipt["input_sha256"]
        )
        groups.setdefault(name.rsplit("-", 1)[0], []).append(digest)
    if any(len(set(digests)) != 1 for digests in groups.values()):
        raise ValueError("Paired methods did not use identical inputs")


def validate_receipt(receipt: dict[str, Any], expected_hash: str) -> None:
    if receipt.get("status") != "measured":
        raise ValueError("Child did not complete a valid measurement")
    gates = receipt["records"] if "records" in receipt else [receipt]
    if not gates or not all(record.get("accuracy_passed") is True for record in gates):
        raise ValueError("Child did not pass every accuracy gate")
    if receipt.get("source_and_tool_sha256") != expected_hash:
        raise ValueError("Source/tool snapshot changed during the run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "baseline"], default="smoke")
    parser.add_argument(
        "--gpu", action="store_true", help="require genuine CuPy runs too"
    )
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--case-timeout", type=float, default=60)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwritten"
    )
    args = parser.parse_args()
    if not (0 < args.budget_seconds <= 600 and 0 < args.case_timeout <= 60):
        parser.error("budget must be in (0,600]; case timeout must be in (0,60]")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    cases = build_cases(args.profile, args.gpu)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    manifest: dict[str, Any] = {
        **metadata(Path(__file__).name),
        "protocol": "numerical-v1",
        "profile": args.profile,
        "status": "running",
        "budget_seconds": args.budget_seconds,
        "case_timeout_seconds": args.case_timeout,
        "child_thread_limits": {
            name: environment[name]
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "unrun_backends": [] if args.gpu else ["cupy"],
        "cases": [{"name": name, "arguments": command} for name, command in cases],
        "completed": [],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    results = {}
    try:
        for name, command in cases:
            timeout = min(
                args.case_timeout, args.budget_seconds - (time.monotonic() - started)
            )
            if timeout <= 0:
                raise TimeoutError("Total benchmark budget exhausted")
            print(f"Running {name}", flush=True)
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        *command,
                        "--output",
                        str(output / f"{name}.json"),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
            receipt = json.loads((output / f"{name}.json").read_text())
            validate_receipt(receipt, manifest["source_and_tool_sha256"])
            results[name] = receipt
            check_pairs(results)
            manifest["completed"].append(name)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["status"] = "smoke_passed" if args.profile == "smoke" else "measured"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise SystemExit(1) from error
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_result(manifest, manifest_path)


if __name__ == "__main__":
    main()
