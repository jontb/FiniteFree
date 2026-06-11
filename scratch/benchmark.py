import time
from typing import List

import numpy as np
import sympy as sp

from finitefree.convolutions import (
    asymmetric_additive,
    multiplicative,
    symmetric_additive,
)
from finitefree.core import RealRootedPolynomial
from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial


def run_benchmarks() -> None:
    print("====================================================================================")
    print("                         FiniteFree Computational Benchmark                         ")
    print("====================================================================================\n")

    degrees: List[int] = [10, 50, 100, 200, 500]

    header = (
        f"{'Degree (d)':<10} | "
        f"{'Init (Lazy) [s]':<15} | "
        f"{'Init (Eager) [s]':<16} | "
        f"{'Sym Add [s]':<12} | "
        f"{'Asym Add [s]':<12} | "
        f"{'Mul [s]':<10} | "
        f"{'Shift [s]':<10}"
    )
    print(header)
    print("-" * 105)

    for d in degrees:
        # Create standard coefficients of length d+1 (monic x^d)
        coeffs: List[float] = [1.0] + [0.0] * d

        # 1. Benchmark Lazy Initialization
        t0 = time.perf_counter()
        RealRootedPolynomial(coeffs, assume_real_rooted=True)
        t_init_lazy = time.perf_counter() - t0

        # 2. Benchmark Eager Initialization (Sturm sequence validation)
        poly_eager = RealRootedPolynomial(coeffs, assume_real_rooted=False)
        t0 = time.perf_counter()
        poly_eager.verify_real_rootedness()
        t_init_eager = time.perf_counter() - t0
        t_eager_str = f"{t_init_eager:.6f}"

        # 3. Benchmark Additive Convolutions
        p = RealRootedPolynomial(coeffs, assume_real_rooted=True)
        q = RealRootedPolynomial(coeffs, assume_real_rooted=True)

        t0 = time.perf_counter()
        symmetric_additive(p, q, d)
        t_sym_add = time.perf_counter() - t0

        t0 = time.perf_counter()
        asymmetric_additive(p, q, d)
        t_asym_add = time.perf_counter() - t0

        # 4. Benchmark Multiplicative Convolution
        t0 = time.perf_counter()
        multiplicative(p, q, d)
        t_mul = time.perf_counter() - t0

        # 5. Benchmark Shift (composed shift)
        t0 = time.perf_counter()
        p.shift(sp.Rational(5, 2))
        t_shift = time.perf_counter() - t0

        print(
            f"{d:<10} | "
            f"{t_init_lazy:<15.6f} | "
            f"{t_eager_str:<16} | "
            f"{t_sym_add:<12.6f} | "
            f"{t_asym_add:<12.6f} | "
            f"{t_mul:<10.6f} | "
            f"{t_shift:<10.6f}"
        )

    print("\n====================================================================================")
    print("                     Multivariate Matrix Pencil Benchmark                           ")
    print("====================================================================================\n")

    m = 3  # variables
    header_pencil = (
        f"{'Dimension (n)':<14} | "
        f"{'Sequential [s]':<18} | "
        f"{'Parallel [s]':<18}"
    )
    print(header_pencil)
    print("-" * 58)

    # Benchmark modular matrix pencil determinant interpolation
    np.random.seed(42)
    for n in [3, 4, 5]:
        matrices = []
        for _ in range(m):
            A = np.random.randint(-5, 5, size=(n, n)).astype(float)
            A = A + A.T  # symmetric
            matrices.append(A)
        pencil = SymmetricMatrixPencil(matrices)

        # 1. Sequential construction
        t0 = time.perf_counter()
        MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(pencil, parallel=False)
        t_seq = time.perf_counter() - t0

        # 2. Parallel construction
        t0 = time.perf_counter()
        MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(pencil, parallel=True)
        t_par = time.perf_counter() - t0

        print(
            f"{n:<14} | "
            f"{t_seq:<18.6f} | "
            f"{t_par:<18.6f}"
        )

    print("\n====================================================================================")
    print("Performance Insights:")
    print(" - Lazy evaluation ensures polynomial creation is O(1).")
    print(" - Convolutions scale in O(d log d) via C-level fmpq_poly multiplication.")
    print(" - Shift operations bypass SymPy symbolics and compose exactly using C-level composition.")
    print(" - Matrix pencils reconstruct determinants mod p, using cached Vandermonde lookups.")
    print("====================================================================================")


if __name__ == "__main__":
    run_benchmarks()
