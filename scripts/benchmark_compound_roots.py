"""Benchmark validation plus uncached compound-Wishart root extraction.

Run with PYTHONPATH pointing at the checkout being measured. The independent
Arb reference uses twice the requested working precision. Construction, public
root calls, and repeat-cache latency are reported separately. Root timings
include domain validation, excluding construction and library warm-up.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import flint
import numpy as np
import scipy
from scipy.linalg import matrix_balance

import finitefree
from finitefree import PrecisionContext, RealRootedPolynomial, laguerre_polynomial


class CountingPolynomial:
    def __init__(self, polynomial: Any) -> None:
        self.polynomial = polynomial
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.polynomial, name)

    def __getitem__(self, index: int) -> Any:
        return self.polynomial[index]

    def complex_roots(self) -> Any:
        self.calls += 1
        return self.polynomial.complex_roots()


def compound_wishart(degree: int) -> RealRootedPolynomial:
    n = degree**2
    q = laguerre_polynomial(degree, n - degree).dilation(flint.fmpq(1, n))
    # degree identical multiplicative convolutions, computed on coefficients.
    return RealRootedPolynomial.from_normalized_coeffs(
        [value**degree for value in q._normalized_coeffs_flint(degree)]
    )


def benchmark(degrees: list[int], precision: int, repeats: int) -> dict[str, Any]:
    matrix_balance(np.eye(2))
    np.linalg.eigvals(np.eye(2))
    records = []
    for degree in degrees:
        started = perf_counter()
        p = compound_wishart(degree)
        construction = perf_counter() - started
        with PrecisionContext(degree=degree, prec=2 * precision):
            started = perf_counter()
            pairs: Any = p._fmpq_poly.complex_roots()
            reference_seconds = perf_counter() - started
            if any(not root.imag.is_zero() for root, _ in pairs):
                raise ValueError("Compound reference contains complex roots")
            bits = min(root.real.rel_accuracy_bits() for root, _ in pairs)
            if bits < 80:
                raise ValueError(f"Insufficient reference accuracy: {bits} bits")
            reference = np.sort(
                [float(root.real) for root, mult in pairs for _ in range(mult)]
            )
        for exact in (False, True):
            trials = []
            for _ in range(repeats):
                fresh = RealRootedPolynomial(p._fmpq_poly)
                probe = CountingPolynomial(fresh._fmpq_poly)
                cast(Any, fresh)._fmpq_poly = probe
                with PrecisionContext(degree=degree, prec=precision):
                    started = perf_counter()
                    roots = fresh.evaluate_roots_float64(exact=exact)
                    seconds = perf_counter() - started
                    started = perf_counter()
                    fresh.evaluate_roots_float64(exact=exact)
                    cached_seconds = perf_counter() - started
                trials.append((seconds, probe.calls, cached_seconds))
            seconds, calls, cached_seconds = min(trials)
            errors = np.abs(roots - reference)
            record = {
                "degree": degree,
                "exact": exact,
                "construction_seconds": construction,
                "reference_seconds": reference_seconds,
                "public_root_seconds": seconds,
                "cached_seconds": cached_seconds,
                "isolation_calls": calls,
                "max_absolute_error": float(np.max(errors)),
                "max_scaled_error": float(np.max(errors))
                / max(1.0, float(np.max(np.abs(reference)))),
                "max_relative_error": float(np.max(errors / np.abs(reference))),
                "reference_accuracy_bits": bits,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    return {
        "module": finitefree.__file__,
        "versions": {
            "flint": flint.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "working_precision_bits": precision,
        "reference_precision_bits": 2 * precision,
        "repeats": repeats,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--degrees", nargs="+", type=int, default=[10, 30, 60, 100, 150]
    )
    parser.add_argument("--precision", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.degrees) < 2 or args.repeats < 1 or args.precision < 80:
        parser.error("degrees must be >=2, repeats >=1 and precision >=80")
    results = benchmark(args.degrees, args.precision, args.repeats)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
