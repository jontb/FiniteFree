"""Compare shifted finite cumulants with analytic Hermite/Wishart contracts.

Run with PYTHONPATH pointing at the checkout being measured. Transform times
are the best of repeated calls with normalized coefficients already cached;
construction is separate. No polynomial root isolation is involved.
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import flint
import numpy as np

import finitefree
from finitefree import FiniteRTransform, gue_expected_poly, wishart_expected_poly


def benchmark(
    degrees: list[int], order: int, precision: int, repeats: int, shift: int
) -> dict[str, Any]:
    records = []
    for family in ["Hermite", "Wishart"]:
        for degree in degrees:
            started = perf_counter()
            p = (
                gue_expected_poly(degree)
                if family == "Hermite"
                else wishart_expected_poly(degree, 2 * degree)
            ).shift(shift)
            construction = perf_counter() - started
            p._normalized_coeffs_flint()
            expected = (
                [0.0, 1.0] + [0.0] * (order - 2)
                if family == "Hermite"
                else [2.0**-k for k in range(order)]
            )
            expected[0] += shift
            times = []
            for _ in range(repeats):
                started = perf_counter()
                actual = FiniteRTransform(
                    p, order=order, numerical=True, prec=precision
                )
                times.append(perf_counter() - started)
            # The large mean cannot assess stability of higher cumulants.
            errors = [abs(a - b) for a, b in zip(actual[1:], expected[1:])]
            max_error = max(errors)
            record = {
                "family": family,
                "degree": degree,
                "construction_seconds": construction,
                "transform_seconds": min(times),
                "nonfinite_outputs": sum(not math.isfinite(x) for x in actual),
                "max_absolute_higher_cumulant_error": (
                    max_error if math.isfinite(max_error) else None
                ),
            }
            records.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
    return {
        "module": finitefree.__file__,
        "versions": {"flint": flint.__version__, "numpy": np.__version__},
        "order": order,
        "precision_bits": precision,
        "repeats": repeats,
        "shift": str(shift),
        "timing_scope": "cached normalized coefficients; construction excluded",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", nargs="+", type=int, default=[60, 150, 300])
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--precision", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shift", type=int, default=10**50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (2 <= args.order <= min(args.degrees)):
        parser.error("order must be >=2 and no larger than any requested degree")
    if args.precision < 64 or args.repeats < 1:
        parser.error("precision must be >=64 and repeats must be >=1")
    result = benchmark(
        args.degrees, args.order, args.precision, args.repeats, args.shift
    )
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
