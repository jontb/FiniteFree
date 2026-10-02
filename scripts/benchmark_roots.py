"""Compare uncached stable roots against independent high-precision Arb roots.

Run from the checkout: PYTHONPATH=. python scripts/benchmark_roots.py --output results.json
Construction and root extraction timings are reported separately; root timings
exclude cached results. No wall-clock thresholds are imposed on CI.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import flint
import numpy as np
import scipy
import sympy as sp

from finitefree import (
    PrecisionContext,
    RealRootedPolynomial,
    gue_expected_poly,
    jacobi_polynomial,
    laguerre_polynomial,
    legendre_polynomial,
    wishart_expected_poly,
)
from finitefree.convolutions import symmetric_additive


def benchmark(degrees: list[int], precision: int) -> dict[str, Any]:
    cases: dict[str, Callable[[int], RealRootedPolynomial]] = {
        "wigner_convolution": lambda n: symmetric_additive(
            gue_expected_poly(n), gue_expected_poly(n), n
        ),
        "wishart_square": lambda n: wishart_expected_poly(n, n),
        "wishart_rectangular": lambda n: wishart_expected_poly(n, 2 * n),
        "legendre": legendre_polynomial,
        "laguerre_hard_edge": lambda n: laguerre_polynomial(
            n, -1 + sp.Rational(1, 10**12)
        ),
        "jacobi_hard_edge": lambda n: jacobi_polynomial(
            n, -1 + sp.Rational(1, 10**12), sp.Rational(3, 2)
        ),
    }
    records = []
    for degree in degrees:
        for name, factory in cases.items():
            started = perf_counter()
            p = factory(degree)
            construction = perf_counter() - started
            with PrecisionContext(degree=degree, prec=precision):
                started = perf_counter()
                pairs: Any = p._fmpq_poly.complex_roots()
                reference_seconds = perf_counter() - started
                if any(not root.imag.is_zero() for root, _ in pairs):
                    raise ValueError("Reference contains nonreal roots")
                bits = min(
                    root.real.rel_accuracy_bits()
                    for root, _ in pairs
                    if not root.real.is_zero()
                )
                if bits < 80:
                    raise ValueError(f"Reference accuracy insufficient: {bits} bits")
                reference = np.sort(
                    [float(root.real) for root, mult in pairs for _ in range(mult)]
                )
            # Warm the numerical library once, then time fresh extractions.
            p.evaluate_roots_float64(exact=False)
            timings = []
            for _ in range(5):
                p._roots_cached = None
                started = perf_counter()
                roots = p.evaluate_roots_float64(exact=False)
                timings.append(perf_counter() - started)
            error = float(np.max(np.abs(roots - reference)))
            record = {
                "case": name,
                "degree": degree,
                "construction_seconds": construction,
                "reference_seconds": reference_seconds,
                "numerical_seconds": min(timings),
                "speedup": reference_seconds / min(timings),
                "max_absolute_error": error,
                "max_scaled_error": error / max(1.0, float(np.max(np.abs(reference)))),
                "smallest_root": float(reference[0]),
                "smallest_root_relative_error": float(
                    abs((roots[0] - reference[0]) / reference[0])
                ),
                "reference_accuracy_bits": bits,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    return {
        "versions": {
            "flint": flint.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "precision_bits": precision,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", nargs="+", type=int, default=[32, 100, 300])
    parser.add_argument("--precision", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = benchmark(args.degrees, args.precision)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
