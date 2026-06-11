import time

import numpy as np

from finitefree.hyperbolic import SymmetricMatrixPencil


def run_benchmark():
    print("=== Parallel Scheduler Benchmark ===")

    # 1. Trivial dimensions (n=3, m=2) - Overhead auto-tuning fallback should bypass executor
    A1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    A2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    pencil_small = SymmetricMatrixPencil([A1, A2])

    print("\nSmall Dimension (n=3, m=2) - Overhead Auto-tuning Check:")
    for backend in ("sequential", "threads", "processes"):
        t0 = time.perf_counter()
        # Evaluate 5 times to compute average
        for _ in range(5):
            pencil_small.diagonal_specialization([1, 1], [2, 3], parallel=True, backend=backend)
        duration = time.perf_counter() - t0
        print(f"  Backend: {backend:<12} | Time: {duration * 1000:7.2f} ms")

    # 2. Large dimensions (n=40, m=2) - ThreadPool should show its strength here
    # 40x40 matrix evaluations are much heavier
    A_large = [np.eye(40), np.fliplr(np.eye(40))]
    pencil_large = SymmetricMatrixPencil(A_large)

    print("\nLarge Dimension (n=40, m=2):")
    for backend in ("sequential", "threads", "processes"):
        t0 = time.perf_counter()
        # Evaluate 3 times
        for _ in range(3):
            pencil_large.diagonal_specialization([1, 1], [2, 3], parallel=True, backend=backend)
        duration = time.perf_counter() - t0
        print(f"  Backend: {backend:<12} | Time: {duration * 1000:7.2f} ms")

if __name__ == "__main__":
    run_benchmark()
