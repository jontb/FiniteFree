"""Profile multivariate APIs against an explicit, local Git baseline.

Run with PYTHONPATH=. and set BLAS/OMP thread counts before starting Python.
The standard suite includes sparse/dense supports, degree/coefficient-size
variation and a 65536-point case. --suite smoke checks the workflow quickly.
"""

import argparse
import cProfile
import functools
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import pstats
import statistics
import subprocess
import sys
import tracemalloc
import types
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import flint
import numpy as np
import sympy as sp

from finitefree.multivariate import MultivariatePolynomial


def load_baseline(root: Path, ref: str) -> tuple[Any, dict[str, str]]:
    """Resolve a local commit; execute only its explicit multivariate module."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=root, text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", f"{commit}:finitefree/multivariate.py"], cwd=root
    )
    module = types.ModuleType("finitefree._multivariate_benchmark_baseline")
    exec(compile(source, f"baseline-{commit}/multivariate.py", "exec"), module.__dict__)
    return module.MultivariatePolynomial, {
        "commit": commit,
        "multivariate_source_sha256": hashlib.sha256(source).hexdigest(),
    }


def make_terms(
    m: int, degree: int, count: int, support: str, bits: int
) -> dict[tuple[int, ...], sp.Rational]:
    rng = np.random.default_rng(20261007 + m * 100 + degree)
    exponents: set[tuple[int, ...]] = set()
    if support == "dense":
        exponents = {
            alpha
            for alpha in itertools.product(range(degree + 1), repeat=m)
            if sum(alpha) <= degree
        }
    else:
        available = sum(math.comb(m, k) * degree**k for k in range(1, min(m, 3) + 1))
        if count > available:
            raise ValueError("Sparse term count exceeds the available support.")
        while len(exponents) < count:
            alpha_list = [0] * m
            for i in rng.choice(m, int(rng.integers(1, min(m, 3) + 1)), replace=False):
                alpha_list[int(i)] = int(rng.integers(1, degree + 1))
            exponents.add(tuple(alpha_list))
    terms = {}
    for alpha in sorted(exponents):
        sign = (-1) ** int(rng.integers(0, 2))
        numerator = int(rng.integers(1, 10))
        denominator = int(rng.integers(1, 8))
        if bits > 16:
            numerator += 2 ** (bits - 1)
            denominator += 2 ** (bits - 1) + 7
        terms[alpha] = sp.Rational(sign * numerator, denominator)
    return terms


def independent_value(
    terms: dict[tuple[int, ...], sp.Rational], point: Any, orders: tuple[int, ...]
) -> float:
    """Direct scalar rational differentiation, independent of library caches."""
    summands = []
    for alpha, coefficient in terms.items():
        value = coefficient
        for exponent, order in zip(alpha, orders):
            if exponent < order:
                value = sp.Rational(0)
                break
            value *= math.prod(range(exponent - order + 1, exponent + 1))
        if value:
            summands.append(
                float(value)
                * math.prod(
                    float(x) ** (e - d) for x, e, d in zip(point, alpha, orders)
                )
            )
    return math.fsum(summands)


def validate_numeric(
    p: Any, terms: dict[tuple[int, ...], sp.Rational], points: Any
) -> None:
    m = len(p.variables)
    for point in points[:3]:
        g = []
        h = []
        for i in range(m):
            order = [0] * m
            order[i] = 1
            g.append(independent_value(terms, point, tuple(order)))
            row = []
            for j in range(m):
                order[j] += 1
                row.append(independent_value(terms, point, tuple(order)))
                order[j] -= 1
            h.append(row)
        for actual, expected in (
            (p.evaluate_float64(point), independent_value(terms, point, (0,) * m)),
            (p.gradient_float64(point), g),
            (p.hessian_float64(point), h),
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def allocations(method: Callable[[], Any]) -> dict[str, int]:
    tracemalloc.start()
    result = method()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "retained_traced_bytes": current,
        "peak_traced_bytes": peak,
        "output_array_bytes": result.nbytes if isinstance(result, np.ndarray) else 0,
    }


def profile(method: Callable[[], Any], calls: int = 5) -> dict[str, Any]:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(calls):
        method()
    profiler.disable()
    stats: Any = pstats.Stats(profiler)
    top = sorted(stats.stats.items(), key=lambda pair: pair[1][2], reverse=True)[:15]
    return {
        "calls": calls,
        "total_calls": stats.total_calls,
        "total_seconds": stats.total_tt,
        "functions": [
            {
                "file": key[0],
                "line": key[1],
                "function": key[2],
                "primitive_calls": value[0],
                "calls": value[1],
                "internal_seconds": value[2],
                "cumulative_seconds": value[3],
            }
            for key, value in top
        ],
    }


def rss_worker(root: Path, ref: str, request: dict[str, Any]) -> dict[str, Any]:
    """Linux/macOS high-water RSS from one fresh interpreter per API/version."""
    import resource

    scale = 1 if sys.platform == "darwin" else 1024

    def peak() -> int:
        if sys.platform == "linux":
            # ru_maxrss may retain a pre-exec parent high water after spawning
            # from a memory-heavy study. VmHWM belongs to this exec's process.
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
            raise RuntimeError("Linux process peak RSS is unavailable.")
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale

    imported = peak()
    baseline_class, _ = load_baseline(root, ref)
    cls = baseline_class if request["version"] == "baseline" else MultivariatePolynomial
    m, degree, support = request["variables"], request["degree"], request["support"]
    variables = sp.symbols(f"x0:{m}")
    terms = make_terms(m, degree, 96 if support == "sparse" else 0, support, 16)
    p = cls.from_coefficients(terms, variables)
    points = np.random.default_rng(11).uniform(-0.6, 0.6, (request["batch"], m))
    method = functools.partial(getattr(p, request["operation"] + "_float64"), points)
    setup_peak = peak()
    started = perf_counter()
    result = method()
    first_seconds = perf_counter() - started
    first_peak, output_bytes = peak(), result.nbytes
    del result
    gc.collect()
    started = perf_counter()
    result = method()
    assert result.nbytes == output_bytes
    return {
        **request,
        "import_peak_rss_bytes": imported,
        "setup_peak_rss_bytes": setup_peak,
        "first_call_peak_rss_bytes": first_peak,
        "warm_call_peak_rss_bytes": peak(),
        "first_call_seconds": first_seconds,
        "warm_call_seconds": perf_counter() - started,
        "input_array_bytes": points.nbytes,
        "output_array_bytes": output_bytes,
        "method": "Fresh serial subprocess. Linux /proc/self/status VmHWM, macOS ru_maxrss. Process high-water RSS includes imports, input/setup, caches and output. Warm peak cannot decrease after the first call; differences are not isolated allocator costs.",
    }


def rss_study(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if (
        sys.platform not in ("linux", "darwin")
        or args.suite == "smoke"
        or not args.large_batch
    ):
        return []
    records = []
    for support, degree in (("sparse", 6), ("dense", 4)):
        for operation in ("evaluate", "gradient", "hessian"):
            for version in ("baseline", "candidate"):
                request = {
                    "variables": 8,
                    "degree": degree,
                    "support": support,
                    "batch": args.large_batch,
                    "operation": operation,
                    "version": version,
                }
                output = subprocess.check_output(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--baseline-ref",
                        args.baseline_ref,
                        "--output",
                        os.devnull,
                        "--rss-worker",
                        json.dumps(request),
                    ],
                    cwd=root,
                    text=True,
                    timeout=120,
                )
                records.append(json.loads(output))
    return records


def measure_pair(
    baseline: Callable[[], Any], candidate: Callable[[], Any], repeats: int, exact: bool
) -> dict[str, Any]:
    before, after = baseline(), candidate()
    if exact:
        assert before.variables == after.variables
        assert before.coefficients() == after.coefficients()
        error = 0.0
    else:
        np.testing.assert_allclose(after, before, rtol=2e-11, atol=2e-11)
        error = float(np.max(np.abs(after - before) / (1 + np.abs(before)), initial=0))
    samples: tuple[list[float], list[float]] = ([], [])
    # Warm both methods. Compare alternating order, outside tracing/profiling.
    for iteration in range(repeats):
        methods = ((baseline, samples[0]), (candidate, samples[1]))
        for method, times in methods if iteration % 2 == 0 else methods[::-1]:
            started = perf_counter()
            method()
            times.append(perf_counter() - started)
    first, second = (statistics.median(sample) for sample in samples)
    return {
        "baseline_seconds": first,
        "candidate_seconds": second,
        "speedup": first / second,
        "baseline_samples_seconds": samples[0],
        "candidate_samples_seconds": samples[1],
        "max_scaled_difference": error,
        "baseline_allocations": allocations(baseline),
        "candidate_allocations": allocations(candidate),
    }


def benchmark(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    baseline_class, baseline_metadata = load_baseline(root, args.baseline_ref)
    families = [
        (3, 0, 0, "dense", 16),
        (8, 1, 0, "dense", 16),
        (3, 6, 96, "sparse", 16),
        (8, 6, 96, "sparse", 16),
        (16, 6, 96, "sparse", 16),
        (8, 2, 96, "sparse", 16),
        (8, 12, 96, "sparse", 16),
        (16, 6, 384, "sparse", 16),
        (3, 6, 0, "dense", 16),
        (8, 4, 0, "dense", 16),
        (3, 6, 0, "dense", 256),
        (8, 6, 96, "sparse", 256),
    ]
    if args.suite == "smoke":
        families = [(3, 2, 12, "sparse", 16), (3, 2, 0, "dense", 256)]
    setup, numeric, contexts, profiles = [], [], [], []
    for m, degree, count, support, bits in families:
        variables = sp.symbols(f"x0:{m}")
        terms = make_terms(m, degree, count, support, bits)
        family = {
            "variables": m,
            "degree_limit": degree,
            "support": support,
            "terms": len(terms),
            "maximum_numerator_bits": max(
                abs(int(c.p)).bit_length() for c in terms.values()
            ),
            "maximum_denominator_bits": max(
                int(c.q).bit_length() for c in terms.values()
            ),
        }
        started = perf_counter()
        before = baseline_class.from_coefficients(terms, variables)
        after = MultivariatePolynomial.from_coefficients(terms, variables)
        construction = perf_counter() - started
        assert before.coefficients() == after.coefficients() == terms
        for name in ("evaluate", "gradient", "hessian"):
            times = []
            for p in (before, after):
                started = perf_counter()
                getattr(p, name + "_float64")(np.zeros(m))
                times.append(perf_counter() - started)
            setup.append(
                {
                    **family,
                    "operation": name,
                    "construction_pair_seconds": construction,
                    "baseline_first_object_call_seconds": times[0],
                    "candidate_first_object_call_seconds": times[1],
                }
            )
        batches = [0, 1, 256, 4096] if args.suite == "standard" else [0, 1, 32]
        if (m, degree, count, support, bits) in (
            (8, 6, 96, "sparse", 16),
            (16, 6, 96, "sparse", 16),
            (8, 4, 0, "dense", 16),
        ) and args.large_batch:
            batches.append(args.large_batch)
        for batch in batches:
            points = np.random.default_rng(1849 + batch).uniform(-0.6, 0.6, (batch, m))
            validate_numeric(after, terms, points)
            for name in ("evaluate", "gradient", "hessian"):
                baseline = functools.partial(getattr(before, name + "_float64"), points)
                candidate = functools.partial(getattr(after, name + "_float64"), points)
                record = {
                    **family,
                    "batch": batch,
                    "layout": "contiguous_batch",
                    "operation": name,
                    **measure_pair(baseline, candidate, args.repeats, False),
                }
                numeric.append(record)
                print(json.dumps(record, allow_nan=False), flush=True)
                if (
                    m == 8
                    and bits == 16
                    and batch == 4096
                    and (degree, support) in ((6, "sparse"), (4, "dense"))
                ):
                    profiles.append(
                        {
                            "operation": name,
                            "support": support,
                            "degree": degree,
                            "baseline": profile(baseline),
                            "candidate": profile(candidate),
                        }
                    )
        if m == 8 and degree == 6 and bits == 16 and args.suite == "standard":
            points = np.random.default_rng(343).uniform(-0.6, 0.6, (5, 2117, m))
            layouts = {
                "single_point": points[0, 0],
                "fortran": np.asfortranarray(points),
                "negative_strides": points[::-1, ::-1, ::-1],
                "transpose": points.transpose(1, 0, 2),
                "broadcast": np.broadcast_to(points[0, :1], points.shape),
            }
            for layout, supplied in layouts.items():
                validate_numeric(after, terms, supplied.reshape(-1, m)[:3])
                for name in ("evaluate", "gradient", "hessian"):
                    record = {
                        **family,
                        "batch": math.prod(supplied.shape[:-1]),
                        "layout": layout,
                        "operation": name,
                        **measure_pair(
                            functools.partial(
                                getattr(before, name + "_float64"), supplied
                            ),
                            functools.partial(
                                getattr(after, name + "_float64"), supplied
                            ),
                            args.repeats,
                            False,
                        ),
                    }
                    numeric.append(record)
                    print(json.dumps(record, allow_nan=False), flush=True)
        renamed = sp.symbols(f"u0:{m}")
        operations: dict[str, tuple[str, Any, Any]] = {
            "reorder_reverse": ("reorder_variables", list(reversed(variables)), None),
            "reorder_identity": ("reorder_variables", list(variables), None),
            "substitute_identity": ("substitute", {}, None),
            "substitute_rename": ("substitute", dict(zip(variables, renamed)), renamed),
            "substitute_permutation": (
                "substitute",
                dict(zip(variables, reversed(variables))),
                None,
            ),
            "substitute_scalar": (
                "substitute",
                {variables[0]: sp.Rational(2, 3)},
                None,
            ),
            "substitute_expanding": (
                "substitute",
                {variables[0]: variables[0] + variables[1] / 3},
                None,
            ),
        }
        if (m, degree, support, bits) in ((8, 6, "sparse", 16), (3, 6, "dense", 16)):
            operations["substitute_all_shift"] = (
                "substitute",
                {v: v + sp.Rational(1, 3) for v in variables},
                None,
            )
        expression = after.expr
        for name, (method, argument, target) in operations.items():
            kwargs = {} if method == "reorder_variables" else {"variables": target}
            baseline = functools.partial(getattr(before, method), argument, **kwargs)
            candidate = functools.partial(getattr(after, method), argument, **kwargs)
            result = candidate()
            started = perf_counter()
            expected = sp.Poly(
                expression
                if method == "reorder_variables"
                else expression.subs(argument, simultaneous=True),
                result.variables,
            ).as_dict()
            assert result.coefficients() == expected
            reference_seconds = perf_counter() - started
            record = {
                **family,
                "operation": name,
                "output_terms": len(expected),
                "output_maximum_numerator_bits": max(
                    (abs(int(c.p)).bit_length() for c in expected.values()), default=0
                ),
                "output_maximum_denominator_bits": max(
                    (int(c.q).bit_length() for c in expected.values()), default=0
                ),
                "reference_seconds": reference_seconds,
                **measure_pair(baseline, candidate, args.repeats, True),
            }
            contexts.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
            if m == 8 and degree == 6 and bits == 16:
                profiles.append(
                    {
                        "operation": name,
                        "baseline": profile(baseline),
                        "candidate": profile(candidate),
                    }
                )
    return {
        "baseline": baseline_metadata,
        "candidate_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "candidate_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
        ),
        "candidate_source_sha256": hashlib.sha256(
            (root / "finitefree/multivariate.py").read_bytes()
        ).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "flint": flint.__version__,
        "sympy": sp.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "suite": args.suite,
        "repeats": args.repeats,
        "method": "Alternating interleaved warm calls; medians. Fresh-object setup, references, tracing and profiles excluded. Tracing includes output arrays but omits some native FLINT memory; it is not RSS. Numerical references independently differentiate rational monomials at selected points; contexts use SymPy composition. Degree limit is per active variable for sparse support and total degree for dense support. First object call follows process imports and can include conversion/differentiation; it is not process startup.",
        "setup": setup,
        "numeric": numeric,
        "contexts": contexts,
        "profiles": profiles,
        "process_rss": rss_study(root, args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ref", required=True, help="Trusted local Git commit/ref to compare."
    )
    parser.add_argument("--suite", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--large-batch", type=int, default=65536, help="0 omits the large cases."
    )
    parser.add_argument("--pin-cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rss-worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.large_batch < 0:
        parser.error("Repeats must be positive and large-batch must be nonnegative.")
    if args.pin_cpu:
        if not hasattr(os, "sched_getaffinity"):
            parser.error("CPU affinity is unavailable on this platform.")
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    root = Path(__file__).resolve().parent.parent
    if args.rss_worker:
        print(
            json.dumps(rss_worker(root, args.baseline_ref, json.loads(args.rss_worker)))
        )
        return
    result = benchmark(root, args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
