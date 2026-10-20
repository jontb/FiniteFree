"""Shared receipts for numerical benchmarks; no library behavior is changed."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flint
import numpy as np
import scipy
import sympy

ROOT = Path(__file__).resolve().parent.parent
SEED = 862


def array_digest(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def metadata(script: str) -> dict[str, Any]:
    paths = sorted((ROOT / "finitefree").rglob("*.py"))
    paths += sorted((ROOT / "finitefree").rglob("*.pyx"))
    paths += sorted((ROOT / "scripts").glob("benchmark_*.py"))
    paths += [
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_numerical_benchmarks.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": script,
        "seed": SEED,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).splitlines(),
        "source_and_tool_sha256": digest.hexdigest(),
        "platform": platform.platform(),
        "cpu": cpu,
        "visible_cpu_count": os.cpu_count(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "flint": flint.__version__,
            "sympy": sympy.__version__,
        },
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def scaled_error(actual: Any, expected: Any) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not (
        np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    ):
        raise ValueError("Nonfinite or differently shaped benchmark/reference output")
    return float(np.max(np.abs(actual - expected) / (1 + np.abs(expected)), initial=0))


def write_result(result: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
