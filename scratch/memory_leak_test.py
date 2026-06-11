import os
import gc
import numpy as np
import sympy as sp
import flint
from finitefree.core import RealRootedPolynomial
from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.convolutions import symmetric_additive

def get_rss_kb() -> int:
    """Reads the current process RSS memory in Kilobytes from /proc."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0

def run_stress():
    print("Starting PyFFP Memory Leak Stress Test...")
    gc.collect()
    start_mem = get_rss_kb()
    print(f"Initial VmRSS: {start_mem} KB")

    # Construct matrices for a pencil
    n, m = 4, 3
    matrices = [np.random.randint(-3, 4, size=(n, n)).astype(float) for _ in range(m)]
    matrices = [A + A.T for A in matrices]
    
    # 1. Warm-up iterations to fill caches
    for _ in range(50):
        pencil = SymmetricMatrixPencil(matrices)
        p = pencil.diagonal_specialization([1, 2, 3], [0, 1, 1])
        q = RealRootedPolynomial.from_roots([1, 2, 3, 4])
        symmetric_additive(p, q, d=4)

    gc.collect()
    post_warmup_mem = get_rss_kb()
    print(f"Post-Warmup VmRSS (Caches Populated): {post_warmup_mem} KB")

    # 2. Measurement iterations
    iterations = 500
    for i in range(iterations):
        pencil = SymmetricMatrixPencil(matrices)
        p = pencil.diagonal_specialization([1, 2, 3], [0, 1, 1])
        q = RealRootedPolynomial.from_roots([1, 2, 3, 4])
        symmetric_additive(p, q, d=4)

    gc.collect()
    end_mem = get_rss_kb()
    print(f"Final VmRSS after {iterations} iterations: {end_mem} KB")
    
    diff = end_mem - post_warmup_mem
    print(f"Memory Difference: {diff} KB")
    
    # If there is a leak, the VmRSS would grow by several megabytes (thousands of KB)
    # A small difference (e.g. less than 500 KB due to python/OS memory allocator fragmentation) is normal.
    if diff > 1000:
        print("WARNING: Potential memory leak detected.")
        exit(1)
    else:
        print("SUCCESS: Memory usage is stable. No leaks detected.")
        exit(0)

if __name__ == "__main__":
    run_stress()
