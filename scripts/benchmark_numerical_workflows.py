"""Matched prepared/scalar kernels and streaming/retained summaries.

Run one method and input size per fresh process. Timing excludes independent
references; a separate traced allocation pass excludes imports/preparation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp
from scipy.special import eval_hermitenorm

from finitefree import (
    EmpiricalComparison,
    OrthogonalPolynomialKernel,
    StreamingEmpiricalComparison,
    gue_expected_poly,
    hermite_polynomial,
    sample_gue,
)
from scripts._benchmark import SEED, array_digest, metadata, scaled_error, write_result
from scripts.benchmark_prepared_pencils import process_peak_rss_bytes

INTERVALS = [(-1.0, 1.0), (0.0, 2.0)]


def query_measurements(
    call: Callable[[], Any],
    repeats: int,
    before_memory: Callable[[], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    actual = call()
    first = time.perf_counter() - started
    del actual
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        actual = call()
        times.append(time.perf_counter() - started)
        del actual
    gc.collect()
    if before_memory is not None:
        before_memory()
    tracemalloc.start()
    try:
        actual = call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return actual, {
        "first_query_seconds": first,
        "warm_query_samples_seconds": times,
        "python_traced_query_peak_bytes": peak,
        "process_peak_rss_before_reference_bytes": process_peak_rss_bytes(),
    }


def benchmark_kernel(
    method: str, rank: int, points: int, repeats: int
) -> dict[str, Any]:
    started = time.perf_counter()
    kernel = OrthogonalPolynomialKernel(
        [hermite_polynomial(j, physicist=False) for j in range(rank + 1)],
        [math.factorial(j) for j in range(rank)],
    )
    construction = time.perf_counter() - started
    xs = np.linspace(-1.25, 1.25, points)
    if points > 1:
        xs[1] = np.nextafter(xs[0], np.inf)
    preparation = 0.0
    call: Callable[[], Any]
    if method == "prepared":
        started = time.perf_counter()
        prepared = kernel.prepare_numeric()
        preparation = time.perf_counter() - started

        def prepared_call() -> Any:
            return prepared.matrix(xs)

        call = prepared_call
    else:

        def scalar_call() -> Any:
            return np.array(
                [[float(kernel(float(x), float(y))) for y in xs] for x in xs]
            )

        call = scalar_call
    actual, measurements = query_measurements(call, repeats)
    # Independent special-function basis on every point; no library kernel calls.
    basis = np.column_stack(
        [eval_hermitenorm(j, xs) / math.sqrt(math.factorial(j)) for j in range(rank)]
    )
    reference = basis @ basis.T
    error = scaled_error(actual, reference)
    # Exact rational Hermite values, including diagonal/adjacent/far pairs.
    pairs = sorted(
        {
            (0, 0),
            (0, min(1, points - 1)),
            (0, points - 1),
            (points // 2, points // 2),
            (points - 1, points - 1),
        }
    )
    exact_errors = []
    for i, k in pairs:
        x, y = sp.Rational(float(xs[i])), sp.Rational(float(xs[k]))
        exact = sum(
            sp.hermite_prob(j, x) * sp.hermite_prob(j, y) / math.factorial(j)
            for j in range(rank)
        )
        exact_errors.append(scaled_error(actual[i, k], float(exact)))
    passed = max([error, *exact_errors]) <= 2e-11
    return {
        "status": "measured" if passed else "invalid",
        "workflow": "kernel",
        "method": method,
        "rank": rank,
        "points": points,
        "repeats": repeats,
        "input_sha256": array_digest(xs),
        "construction_seconds": construction,
        "preparation_seconds": preparation,
        **measurements,
        "output_storage_bytes": actual.nbytes,
        "independent_basis_scaled_error": error,
        "exact_sentinel_pairs": pairs,
        "exact_sentinel_scaled_errors": exact_errors,
        "accuracy_scaled_tolerance": 2e-11,
        "accuracy_passed": passed,
        "timing_scope": "full points-by-points kernel matrix; construction/preparation and references excluded; repeated calls use the same coordinates",
        "memory_scope": "traced query allocation excludes existing kernel/basis preparation; RSS includes imports and prior calls; neither is total system memory",
    }


def summarize_retained(comparison: EmpiricalComparison) -> dict[str, Any]:
    counts = np.stack(
        [
            np.count_nonzero(
                (comparison.eigenvalues >= a) & (comparison.eigenvalues <= b), axis=1
            )
            for a, b in INTERVALS
        ]
    )
    return {
        "mean": comparison.char_poly_coeffs.mean(axis=0),
        "std": comparison.char_poly_coeffs.std(axis=0, ddof=1),
        "histograms": np.stack(
            [np.bincount(row, minlength=comparison.d + 1) for row in counts]
        ),
    }


def benchmark_summary(
    method: str, degree: int, samples: int, repeats: int
) -> dict[str, Any]:
    polynomial = gue_expected_poly(degree)
    measured_digest = hashlib.sha256()
    record_inputs = False

    def capture_inputs() -> None:
        nonlocal record_inputs
        record_inputs = True

    def call() -> tuple[Any, dict[str, Any]]:
        rng = np.random.default_rng(SEED)

        def generator() -> Any:
            matrix = sample_gue(degree, rng=rng)
            if record_inputs:
                measured_digest.update(np.ascontiguousarray(matrix).tobytes())
            return matrix

        if method == "streaming":
            stream = StreamingEmpiricalComparison(polynomial, INTERVALS)
            stream.extend(generator, samples)
            return stream, {
                "mean": stream.coefficient_mean,
                "std": stream.coefficient_std,
                "histograms": stream.count_histograms,
            }
        retained = EmpiricalComparison(polynomial, samples, generator)
        return retained, summarize_retained(retained)

    (comparison, summary), measurements = query_measurements(
        call, repeats, capture_inputs
    )
    owned_bytes = sum(
        value.nbytes
        for value in vars(comparison).values()
        if isinstance(value, np.ndarray)
    )
    del comparison
    # Independent online reference: the determinant generating identity
    # det(zI-H)=prod(z-lambda), using scalar coefficient convolution (not np.poly).
    # Welford statistics have a separate longdouble implementation here.
    rng = np.random.default_rng(SEED)
    digest = hashlib.sha256()
    mean = np.zeros(degree + 1, dtype=np.longdouble)
    m2 = np.zeros_like(mean)
    histograms = np.zeros((len(INTERVALS), degree + 1), dtype=np.int64)
    for index in range(samples):
        matrix = sample_gue(degree, rng=rng)
        digest.update(np.ascontiguousarray(matrix).tobytes())
        roots = np.linalg.eigvalsh(matrix)
        coefficients = np.ones(1, dtype=np.longdouble)
        for root in roots:
            updated = np.zeros(len(coefficients) + 1, dtype=np.longdouble)
            updated[:-1] += coefficients
            updated[1:] -= np.longdouble(root) * coefficients
            coefficients = updated
        delta = coefficients - mean
        mean += delta / (index + 1)
        m2 += delta * (coefficients - mean)
        for j, (a, b) in enumerate(INTERVALS):
            histograms[j, np.count_nonzero((roots >= a) & (roots <= b))] += 1
    std = np.sqrt(m2 / (samples - 1))
    errors = [scaled_error(summary["mean"], mean), scaled_error(summary["std"], std)]
    inputs_match = measured_digest.hexdigest() == digest.hexdigest()
    passed = (
        inputs_match
        and max(errors) <= 2e-10
        and np.array_equal(summary["histograms"], histograms)
    )
    return {
        "status": "measured" if passed else "invalid",
        "workflow": "summary",
        "method": method,
        "degree": degree,
        "samples": samples,
        "repeats": repeats,
        "input_sha256": measured_digest.hexdigest(),
        "reference_input_sha256": digest.hexdigest(),
        "inputs_match_reference": inputs_match,
        "intervals": INTERVALS,
        **measurements,
        "owned_ndarray_bytes": owned_bytes,
        "coefficient_mean": summary["mean"].tolist(),
        "coefficient_std": summary["std"].tolist(),
        "count_histograms": summary["histograms"].tolist(),
        "independent_mean_std_scaled_errors": errors,
        "accuracy_scaled_tolerance": 2e-10,
        "accuracy_passed": bool(passed),
        "timing_scope": "seeded GUE generation, eigensolves, accumulation and coefficient/count summary; each call resets the RNG; independent reference excluded",
        "memory_scope": "traced allocation is a separate full-workflow pass including one-matrix input-hash temporaries; RSS includes imports and earlier same-sized calls; owned_ndarray_bytes counts direct array fields only, excluding Python/FLINT objects and returned summary copies",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="workflow", required=True)
    kernel = subparsers.add_parser("kernel")
    kernel.add_argument("--method", choices=["prepared", "scalar"], required=True)
    kernel.add_argument("--rank", type=int, default=16)
    kernel.add_argument("--points", type=int, default=16)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--method", choices=["streaming", "retained"], required=True)
    summary.add_argument("--degree", type=int, default=8)
    summary.add_argument("--samples", type=int, default=64)
    for subparser in (kernel, summary):
        subparser.add_argument("--repeats", type=int, default=3)
        subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.workflow == "kernel" and not (1 <= args.rank <= 128 and args.points >= 1):
        parser.error("rank must be 1..128 (reference scope); points must be positive")
    if args.workflow == "summary" and not (args.degree >= 1 and args.samples >= 2):
        parser.error("degree must be positive; samples must be at least two")
    receipt = metadata(Path(__file__).name)
    try:
        result = (
            benchmark_kernel(args.method, args.rank, args.points, args.repeats)
            if args.workflow == "kernel"
            else benchmark_summary(args.method, args.degree, args.samples, args.repeats)
        )
    except Exception as error:
        write_result(
            {
                **receipt,
                "status": "failed",
                "arguments": vars(args) | {"output": str(args.output)},
                "error": f"{type(error).__name__}: {error}",
            },
            args.output,
        )
        raise SystemExit(1) from error
    write_result({**receipt, **result}, args.output)
    if result["status"] != "measured":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
