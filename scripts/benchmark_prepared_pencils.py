"""Measure prepared pencils; GPU measurements require real hardware, never mocks."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import os
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy

from finitefree.hyperbolic import PreparedMatrixPencil
from scripts._benchmark import SEED, array_digest, metadata, scaled_error, write_result


def timed(call: Callable[[], Any], sync: Callable[[], Any]) -> tuple[Any, float]:
    sync()
    start = time.perf_counter()
    result = call()
    sync()
    return result, time.perf_counter() - start


def process_peak_rss_bytes() -> int | None:
    """Native process high-water RSS, including imports/setup and prior cases."""
    try:
        resource = importlib.import_module("resource")
    except ImportError:
        return None
    scale = 1 if platform.system() == "Darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


def independent_reference(a: Any, x: Any, v: Any) -> tuple[Any, ...]:
    """Dense NumPy solves and trace identities, independent of prepared factors."""
    matrices = np.einsum("bi,ijk->bjk", x, a)
    signs, logs = np.linalg.slogdet(matrices)
    gradient = np.empty((len(x), len(a)))
    hvp = np.empty_like(gradient)
    direction = np.einsum("i,ijk->jk", v, a)
    for row, matrix in enumerate(matrices):
        solved_direction = np.linalg.solve(matrix, direction)
        for j, coefficient in enumerate(a):
            solved = np.linalg.solve(matrix, coefficient)
            gradient[row, j] = np.trace(solved)
            hvp[row, j] = -np.trace(solved_direction @ solved)
    return signs, logs, gradient, hvp


def benchmark(
    backend: str, n: int, m: int, batches: list[int], repeats: int
) -> dict[str, Any]:
    initial_rss = process_peak_rss_bytes()
    xp: Any = np
    device = None

    def sync() -> Any:
        return None

    context_seconds = 0.0
    if backend == "cupy":
        start = time.perf_counter()
        xp = importlib.import_module("cupy")
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No real CUDA device available")
        xp.zeros(1).sum().item()
        sync = xp.cuda.get_current_stream().synchronize
        sync()
        context_seconds = time.perf_counter() - start
        props = xp.cuda.runtime.getDeviceProperties(xp.cuda.runtime.getDevice())
        device = str(props["name"])
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=(m, n, n)) / (10 * n * m)
    a += a.transpose(0, 2, 1)
    a[0] = np.eye(n)
    resident_a, coefficient_transfer = timed(lambda: xp.asarray(a), sync)
    prepared, preparation = timed(
        lambda: PreparedMatrixPencil(resident_a, backend=backend), sync
    )
    records = []
    for batch in batches:
        # A case must not change when earlier batches are omitted or reordered.
        rng = np.random.default_rng(np.random.SeedSequence([SEED, n, m, batch]))
        x = rng.uniform(-1, 1, (batch, m))
        x[:, 0] = 2.0
        v = rng.normal(size=m)
        (resident_x, resident_v), transfer = timed(
            lambda x=x, v=v: (xp.asarray(x), xp.asarray(v)),  # type: ignore[misc]
            sync,
        )
        factor, cold_factor = timed(
            functools.partial(prepared.factor, resident_x), sync
        )
        _, cold_hvp = timed(functools.partial(factor.logabsdet_hvp, resident_v), sync)
        factor_samples = []
        hvp_samples = []
        gradient_samples = []
        event_samples = []
        for _ in range(repeats):
            _, elapsed = timed(functools.partial(prepared.factor, resident_x), sync)
            factor_samples.append(elapsed)
            # This reuses LU factors and the packed coefficient matrices.
            _, elapsed = timed(
                functools.partial(factor.logabsdet_hvp, resident_v), sync
            )
            hvp_samples.append(elapsed)
            _, elapsed = timed(factor.logabsdet_gradient, sync)
            gradient_samples.append(elapsed)
            if backend == "cupy":
                begin, end = xp.cuda.Event(), xp.cuda.Event()
                begin.record()
                factor.logabsdet_hvp(resident_v)
                end.record()
                end.synchronize()
                event_samples.append(xp.cuda.get_elapsed_time(begin, end) / 1000)
        actual = (
            *factor.slogdet(),
            factor.logabsdet_gradient(),
            factor.logabsdet_hvp(resident_v),
        )
        host, output_transfer = timed(
            lambda actual=actual: (  # type: ignore[misc]
                tuple(xp.asnumpy(y) for y in actual) if backend == "cupy" else actual
            ),
            sync,
        )
        rss_before_reference = process_peak_rss_bytes()
        reference = PreparedMatrixPencil(a)
        ref = reference.factor(x)
        expected = (*ref.slogdet(), ref.logabsdet_gradient(), ref.logabsdet_hvp(v))
        errors = [scaled_error(g, e) for g, e in zip(host, expected)]
        independent = independent_reference(a, x, v)
        independent_errors = [scaled_error(g, e) for g, e in zip(host, independent)]
        # Independent matrix slogdet and finite-difference derivative checks.
        direct = np.linalg.slogdet(np.einsum("bi,ijk->bjk", x, a))
        log_error = float(np.max(np.abs(host[1] - direct[1])))
        eps = 1e-5
        fd = (
            reference.factor(x + eps * v).logabsdet_gradient()
            - reference.factor(x - eps * v).logabsdet_gradient()
        ) / (2 * eps)
        hvp_error = float(np.max(np.abs(host[3] - fd)))
        tracemalloc.start()
        measured = prepared.factor(resident_x)
        measured.logabsdet_hvp(resident_v)
        sync()
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        device_peak = None
        if backend == "cupy":
            # Isolated allocation pool: total_bytes is its cached high-water
            # allocation, not total device memory (solver workspaces may differ).
            pool = xp.cuda.MemoryPool()
            with xp.cuda.using_allocator(pool.malloc):
                measured = prepared.factor(resident_x)
                measured.logabsdet_hvp(resident_v)
                sync()
                device_peak = pool.total_bytes()
        records.append(
            {
                "batch": batch,
                "input_sha256": array_digest(a, x, v),
                "accuracy_passed": errors[0] == independent_errors[0] == 0.0
                and max(errors + independent_errors) <= 5e-10,
                "independent_scaled_errors_sign_log_gradient_hvp": independent_errors,
                "input_transfer_seconds": transfer if backend == "cupy" else 0.0,
                "output_transfer_seconds": output_transfer
                if backend == "cupy"
                else 0.0,
                "first_factor_seconds": cold_factor,
                "first_hvp_and_gradient_seconds": cold_hvp,
                "warm_factor_samples_seconds": factor_samples,
                "warm_reused_factor_hvp_samples_seconds": hvp_samples,
                "cached_gradient_samples_seconds": gradient_samples,
                "cuda_event_hvp_samples_seconds": event_samples,
                "python_traced_peak_bytes": traced_peak,
                "process_peak_rss_bytes": process_peak_rss_bytes(),
                "process_peak_rss_before_reference_bytes": rss_before_reference,
                "isolated_cupy_pool_high_water_bytes": device_peak,
                "coefficient_storage_bytes": 2 * m * n * n * 8,
                "lu_storage_bytes_excluding_pivots": batch * n * n * 8,
                "cpu_reference_scaled_errors_sign_log_gradient_hvp": errors,
                "independent_slogdet_absolute_error": log_error,
                "finite_difference_hvp_absolute_error": hvp_error,
            }
        )
    root = Path(__file__).resolve().parent.parent
    return {
        **metadata(Path(__file__).name),
        "status": "measured"
        if all(r["accuracy_passed"] for r in records)
        else "invalid",
        "accuracy_scaled_tolerance": 5e-10,
        "backend": backend,
        "device": device,
        "dtype": "float64",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cupy": xp.__version__ if backend == "cupy" else None,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(
            (root / "finitefree/hyperbolic/prepared.py").read_bytes()
        ).hexdigest(),
        "visible_cpu_count": os.cpu_count(),
        "thread_limits": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "matrix_size": n,
        "variables": m,
        "context_initialization_seconds": context_seconds,
        "initial_process_peak_rss_bytes": initial_rss,
        "coefficient_transfer_seconds": coefficient_transfer
        if backend == "cupy"
        else 0.0,
        "preparation_seconds": preparation,
        "memory_note": "Process peak RSS is cumulative, includes imports/setup/prior cases, and is unavailable on platforms without resource. Traced Python memory and isolated CuPy pool high-water are not total device peak memory. Record external device telemetry separately. Outputs, coefficients and existing factors are excluded from the isolated pool.",
        "timing_note": "Wall timings synchronize the current stream. First calls are process/device warm after imports/context initialization; use a fresh process per backend for cold context cost. CUDA events cover reused HVP queries including host submission gaps. CPU records never imply GPU speedups. Compare matching backend runs across batches to locate a crossover.",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--variables", type=int, default=4)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.size, args.variables, args.repeats, *args.batches) < 1:
        parser.error("size, variables, batches, and repeats must be positive")
    try:
        result = benchmark(
            args.backend, args.size, args.variables, args.batches, args.repeats
        )
    except Exception as error:
        result = {
            "status": "failed",
            "backend": args.backend,
            "error": f"{type(error).__name__}: {error}",
        }
        write_result(result, args.output)
        raise SystemExit(1) from error
    write_result(result, args.output)
    if result["status"] != "measured":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
