"""Measure Hermite kernel accuracy near the diagonal against exact basis sums.

PYTHONPATH selects the checkout. Construction/reference evaluation is excluded
from timings; polynomial float-coefficient caches are warmed before measuring.
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import flint
import numpy as np
import sympy as sp

import finitefree
from finitefree import OrthogonalPolynomialKernel, hermite_polynomial


def benchmark(degrees: list[int], repeats: int) -> dict[str, Any]:
    records = []
    for degree in degrees:
        polys = [hermite_polynomial(j, physicist=False) for j in range(degree + 1)]
        norms = [math.factorial(j) for j in range(degree)]
        kernel = OrthogonalPolynomialKernel(polys, norms)
        for p in polys:
            p.evaluate(2.0)
        for case, x, y in [
            ("diagonal", 2.0, 2.0),
            ("nearby", 2.0, 2.000000001),
            ("separated", 0.5, 1.2),
        ]:
            x_exact, y_exact = sp.Rational(x), sp.Rational(y)
            reference = float(
                sum(
                    p.evaluate(x_exact) * p.evaluate(y_exact) / norm
                    for p, norm in zip(polys, norms)
                )
            )
            kernel(x, y)  # Warm the derivative caches as well.
            times = []
            for _ in range(repeats):
                started = perf_counter()
                actual = float(kernel(x, y))
                times.append(perf_counter() - started)
            record = {
                "degree": degree,
                "case": case,
                "seconds": min(times),
                "reference": reference,
                "relative_error": abs(actual - reference) / abs(reference),
            }
            records.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
    return {
        "module": finitefree.__file__,
        "versions": {"flint": flint.__version__, "numpy": np.__version__},
        "repeats": repeats,
        "timing_scope": "warm coefficient caches; construction/reference excluded",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", nargs="+", type=int, default=[20, 40, 60])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.degrees) < 2 or args.repeats < 1:
        parser.error("degrees must be >=2 and repeats must be >=1")
    result = benchmark(args.degrees, args.repeats)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
