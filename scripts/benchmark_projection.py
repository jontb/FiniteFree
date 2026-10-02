"""Time exact projections against independent Hermite/Laguerre derivative laws.

Construction and analytic reference generation are reported separately from the
best-of-repeat public projection timing. PYTHONPATH selects the checkout.
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import flint

import finitefree
from finitefree import gue_expected_poly, hermite_polynomial, wishart_expected_poly


def benchmark(degrees: list[int], dimension: int, repeats: int) -> dict[str, Any]:
    records = []
    for family in ["GUE", "Wishart"]:
        for degree in degrees:
            started = perf_counter()
            source = (
                gue_expected_poly(degree)
                if family == "GUE"
                else wishart_expected_poly(degree, 2 * degree)
            )
            construction = perf_counter() - started
            reference = (
                hermite_polynomial(dimension, physicist=False).dilation(
                    flint.fmpq(1, math.isqrt(degree))
                )
                if family == "GUE"
                else wishart_expected_poly(dimension, 2 * degree)
            )
            times = []
            for _ in range(repeats):
                started = perf_counter()
                projected = source.projection(dimension)
                times.append(perf_counter() - started)
            record = {
                "family": family,
                "degree": degree,
                "dimension": dimension,
                "construction_seconds": construction,
                "projection_seconds": min(times),
                "coefficients_match_reference": list(projected.coeffs)
                == list(reference.coeffs),
                "has_root_recurrence": projected._root_recurrence is not None,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    return {
        "module": finitefree.__file__,
        "flint_version": flint.__version__,
        "repeats": repeats,
        "timing_scope": "projection only; construction and analytic reference excluded",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", nargs="+", type=int, default=[100, 400, 900])
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(d < 1 or math.isqrt(d) ** 2 != d for d in args.degrees):
        parser.error(
            "degrees must be positive perfect squares for the rational reference"
        )
    if not (0 < args.dimension < min(args.degrees)) or args.repeats < 1:
        parser.error("dimension must be positive and below every degree; repeats >=1")
    result = benchmark(args.degrees, args.dimension, args.repeats)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
