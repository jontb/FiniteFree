"""Compare batched derivatives with preconstructed exact component polynomials.

Run PYTHONPATH=. python scripts/benchmark_multivariate.py --pin-cpu --output results.json.
Construction, exact differentiation and first coefficient conversion are reported
separately from interleaved warm-call medians. Traced memory includes Python/NumPy
allocations and outputs, not all native allocations or process RSS.
"""

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import flint
import numpy as np
import sympy as sp

from finitefree.multivariate import MultivariatePolynomial


def component_gradient(components: list[MultivariatePolynomial], points: Any) -> Any:
    return np.stack([g.evaluate_float64(points) for g in components], axis=-1)


def component_hessian(
    components: list[list[MultivariatePolynomial]], points: Any
) -> Any:
    return np.stack(
        [
            np.stack([h.evaluate_float64(points) for h in row], axis=-1)
            for row in components
        ],
        axis=-2,
    )


def symbolic_substitute(
    expression: Any, replacements: Any, variables: Any
) -> MultivariatePolynomial:
    return MultivariatePolynomial(
        expression.subs(replacements, simultaneous=True), variables
    )


def measure_pair(
    baseline: Callable[[], Any], candidate: Callable[[], Any], repeats: int
) -> dict[str, Any]:
    expected, actual = baseline(), candidate()
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    for iteration in range(repeats):
        methods = ((baseline, baseline_times), (candidate, candidate_times))
        if iteration % 2:
            methods = methods[::-1]
        for method, times in methods:
            started = perf_counter()
            method()
            times.append(perf_counter() - started)
    traced = []
    for method in (baseline, candidate):
        tracemalloc.start()
        method()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced.append(peak)
    before, after = (
        statistics.median(baseline_times),
        statistics.median(candidate_times),
    )
    return {
        "baseline_seconds": before,
        "candidate_seconds": after,
        "speedup": before / after,
        "baseline_samples_seconds": baseline_times,
        "candidate_samples_seconds": candidate_times,
        "baseline_peak_traced_bytes": traced[0],
        "candidate_peak_traced_bytes": traced[1],
    }


def make_polynomial(
    variable_count: int, term_count: int, degree: int
) -> MultivariatePolynomial:
    support = sum(
        math.comb(variable_count, k) * degree**k
        for k in range(1, min(variable_count, 3) + 1)
    )
    if term_count > support:
        raise ValueError("Requested terms exceed available sparse benchmark support.")
    rng = np.random.default_rng(732 + variable_count)
    variables = sp.symbols(f"x0:{variable_count}")
    terms: dict[tuple[int, ...], sp.Rational] = {}
    while len(terms) < term_count:
        alpha = [0] * variable_count
        count = int(rng.integers(1, min(variable_count, 3) + 1))
        for i in rng.choice(variable_count, count, replace=False):
            alpha[int(i)] = int(rng.integers(1, degree + 1))
        numerator = int(rng.integers(-9, 10))
        if numerator:
            terms[tuple(alpha)] = sp.Rational(numerator, int(rng.integers(1, 8)))
    return MultivariatePolynomial.from_coefficients(terms, variables)


def benchmark(
    dimensions: list[int], batches: list[int], terms: int, degree: int, repeats: int
) -> dict[str, Any]:
    records = []
    setup = []
    contexts = []
    for m in dimensions:
        started = perf_counter()
        p = make_polynomial(m, terms, degree)
        construction = perf_counter() - started
        started = perf_counter()
        gradient = p.gradient()
        hessian = p.hessian()
        component_setup = perf_counter() - started
        started = perf_counter()
        p.gradient_float64(np.zeros(m))
        p.hessian_float64(np.zeros(m))
        batch_setup = perf_counter() - started
        setup.append(
            {
                "variables": m,
                "terms": len(p.coefficients()),
                "construction_seconds": construction,
                "exact_component_construction_seconds": component_setup,
                "new_derivative_cache_first_call_seconds": batch_setup,
            }
        )
        expression = p.expr
        variables = p.variables
        independent_g = sp.lambdify(
            variables,
            [sp.diff(expression, x) for x in variables],
            modules="numpy",
            cse=True,
        )
        independent_h = sp.lambdify(
            variables, sp.hessian(expression, variables), modules="numpy", cse=True
        )
        for batch in batches:
            points = np.random.default_rng(591 + batch).uniform(-0.7, 0.7, (batch, m))
            actual_g, actual_h = p.gradient_float64(points), p.hessian_float64(points)
            for point in points[:3]:
                symbolic_g = np.asarray(independent_g(*point), dtype=float).reshape(m)
                symbolic_h = np.asarray(independent_h(*point), dtype=float).reshape(
                    m, m
                )
                np.testing.assert_allclose(
                    p.gradient_float64(point), symbolic_g, rtol=3e-12, atol=3e-12
                )
                np.testing.assert_allclose(
                    p.hessian_float64(point), symbolic_h, rtol=3e-12, atol=3e-12
                )
            for operation in ("gradient", "hessian"):
                if operation == "gradient":
                    baseline = functools.partial(component_gradient, gradient, points)
                    candidate = functools.partial(p.gradient_float64, points)
                    actual = actual_g
                else:
                    baseline = functools.partial(component_hessian, hessian, points)
                    candidate = functools.partial(p.hessian_float64, points)
                    actual = actual_h
                expected = baseline()
                relative = float(
                    np.max(np.abs(actual - expected) / (1 + np.abs(expected)))
                )
                record = {
                    "operation": operation,
                    "variables": m,
                    "batch": batch,
                    "terms": len(p.coefficients()),
                    "max_scaled_component_error": relative,
                    **measure_pair(baseline, candidate, repeats),
                }
                records.append(record)
                print(json.dumps(record, allow_nan=False), flush=True)

        order = list(reversed(variables))
        replacements = {x: x + sp.Rational(1, 3) for x in variables}
        started = perf_counter()
        expected_substitution = sp.expand(
            expression.subs(replacements, simultaneous=True)
        )
        symbolic_substitution_setup = perf_counter() - started
        # Native composition is checked against a separately expanded SymPy result.
        assert p.substitute(replacements).expr == expected_substitution
        for operation in ("reorder", "simultaneous_substitution"):
            if operation == "reorder":
                baseline_context = functools.partial(
                    MultivariatePolynomial, expression, order
                )
                candidate_context = functools.partial(p.reorder_variables, order)
                expected_expression = expression
            else:
                baseline_context = functools.partial(
                    symbolic_substitute, expression, replacements, variables
                )
                candidate_context = functools.partial(p.substitute, replacements)
                expected_expression = expected_substitution
            assert (
                baseline_context().expr
                == candidate_context().expr
                == expected_expression
            )
            baseline_samples: list[float] = []
            candidate_samples: list[float] = []
            for iteration in range(repeats):
                methods = (
                    (baseline_context, baseline_samples),
                    (candidate_context, candidate_samples),
                )
                if iteration % 2:
                    methods = methods[::-1]
                for method, times in methods:
                    started = perf_counter()
                    method()
                    times.append(perf_counter() - started)
            before, after = (
                statistics.median(baseline_samples),
                statistics.median(candidate_samples),
            )
            contexts.append(
                {
                    "operation": operation,
                    "variables": m,
                    "terms": len(p.coefficients()),
                    "baseline_seconds": before,
                    "candidate_seconds": after,
                    "speedup": before / after,
                    "baseline_samples_seconds": baseline_samples,
                    "candidate_samples_seconds": candidate_samples,
                    "independent_symbolic_substitution_reference_seconds": symbolic_substitution_setup,
                }
            )
    root = Path(__file__).resolve().parent.parent
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "multivariate_source_sha256": hashlib.sha256(
            (root / "finitefree/multivariate.py").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "flint": flint.__version__,
        "sympy": sp.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "method": "Alternating interleaved warmed calls; median samples. Exact components are preconstructed and coefficients warmed; setup and independent symbolic reference excluded. Traced memory is not RSS.",
        "repeats": repeats,
        "degree_per_active_variable": degree,
        "setup": setup,
        "derivatives": records,
        "contexts": contexts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variables", type=int, nargs="+", default=[3, 8, 16])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 256, 4096])
    parser.add_argument("--terms", type=int, default=96)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pin-cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(
        n < 1
        for n in [*args.variables, *args.batches, args.terms, args.degree, args.repeats]
    ):
        parser.error("Dimensions, batches, terms, degree and repeats must be positive.")
    if args.pin_cpu:
        if not hasattr(os, "sched_getaffinity"):
            parser.error("CPU affinity is unavailable on this platform.")
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    result = benchmark(
        args.variables, args.batches, args.terms, args.degree, args.repeats
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
