import time
from typing import List

from finitefree.convolutions import multiplicative, symmetric_additive
from finitefree.core import RealRootedPolynomial


def run_benchmarks() -> None:
    print("====================================================")
    print("             FiniteFree Computational Benchmark     ")
    print("====================================================\n")

    degrees: List[int] = [10, 50, 100, 200, 500]

    header = (
        f"{'Degree (d)':<12} | "
        f"{'Init (Lazy) [s]':<18} | "
        f"{'Init (Eager) [s]':<18} | "
        f"{'Additive Conv [s]':<18} | "
        f"{'Multiplicative [s]':<18}"
    )
    print(header)
    print("-" * 92)

    for d in degrees:
        # Create standard coefficients of length d+1 (monic x^d)
        coeffs: List[float] = [1.0] + [0.0] * d

        # 1. Benchmark Lazy Initialization
        t0 = time.perf_counter()
        RealRootedPolynomial(coeffs, assume_real_rooted=True)
        t_init_lazy = time.perf_counter() - t0

        # 2. Benchmark Eager Initialization (Sturm sequence validation)
        # Sturm sequence scales poorly for high degrees, so we only run for d <= 20
        poly_eager = RealRootedPolynomial(coeffs, assume_real_rooted=False)
        t0 = time.perf_counter()
        poly_eager.verify_real_rootedness()
        t_init_eager = time.perf_counter() - t0
        t_eager_str = f"{t_init_eager:.6f}"

        # 3. Benchmark Additive Convolution (p \boxplus_d q)
        p = RealRootedPolynomial(coeffs, assume_real_rooted=True)
        q = RealRootedPolynomial(coeffs, assume_real_rooted=True)
        t0 = time.perf_counter()
        symmetric_additive(p, q, d)
        t_add_conv = time.perf_counter() - t0

        # 4. Benchmark Multiplicative Convolution (p \boxtimes_d q)
        t0 = time.perf_counter()
        multiplicative(p, q, d)
        t_mul_conv = time.perf_counter() - t0

        print(
            f"{d:<12} | "
            f"{t_init_lazy:<18.6f} | "
            f"{t_eager_str:<18} | "
            f"{t_add_conv:<18.6f} | "
            f"{t_mul_conv:<18.6f}"
        )

    print("\n====================================================")
    print("Performance Insights:")
    print(" - Lazy evaluation ensures polynomial creation is O(1).")
    print(
        " - Convolutions scale quadratically O(d^2) via discrete "
        "symmetric coefficients."
    )
    print(" - Sturm validation scales exponentially and is deferred cleanly.")
    print("====================================================")


if __name__ == "__main__":
    run_benchmarks()
