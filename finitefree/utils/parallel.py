import concurrent.futures
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import flint
import numpy as np

from .modular import (
    get_inverse_vandermonde_matrices_cached,
    get_inverse_vandermonde_matrix,
    modular_det,
)

try:
    if os.environ.get("PYFFP_DISABLE_CYTHON") == "1":
        raise ImportError("Cython explicitly disabled via environment variable")
    from .modular_fast import (  # type: ignore[import-untyped]
        eval_diagonal_specialization_mod_p,
        eval_points_grid_mod_p,
    )
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def _eval_diagonal_specialization_prime_worker(
    p: int, A_int: Any, B_int: Any, n: int, deg: int
) -> Tuple[int, Optional[List[int]]]:
    if HAS_CYTHON:
        A_arr = np.array(A_int, dtype=np.int64)
        B_arr = np.array(B_int, dtype=np.int64)
        y = list(eval_diagonal_specialization_mod_p(A_arr, B_arr, p, deg))
    else:
        def eval_mod_p(z_val: int, p_val: int) -> int:
            M = [[(z_val * A_int[r][c] + B_int[r][c]) % p_val for c in range(n)] for r in range(n)]
            return int(modular_det(M, p_val))
        y = [eval_mod_p(k, p) for k in range(deg + 1)]

    try:
        V_inv = get_inverse_vandermonde_matrix(deg + 1, p)

        # Offload matrix multiplication to compiled C-level nmod_mat inside Flint
        V_inv_flat = [val for row in V_inv for val in row]
        V_inv_mat = flint.nmod_mat(deg + 1, deg + 1, V_inv_flat, p)
        y_mat = flint.nmod_mat(deg + 1, 1, y, p)
        c_mat = V_inv_mat * y_mat
        c_p = [int(c_mat[r, 0]) for r in range(deg + 1)]
        return p, c_p
    except Exception:
        return p, None


def _eval_prime_worker(
    p: int, integer_matrices: Any, full_grid_pts: Any, n: int, m: int, exps: Any
) -> Tuple[int, Optional[List[int]]]:
    def interpolate_full_grid(
        values: Dict[Tuple[int, ...], int], v: int, d: int, p_val: int, inv_vands: Any
    ) -> Dict[Tuple[int, ...], int]:
        current_values = dict(values)
        for step in range(v):
            from collections import defaultdict
            grouped = defaultdict(list)
            for pt, val in current_values.items():
                vi = pt[step]
                rem = pt[:step] + pt[step + 1 :]
                grouped[rem].append((vi, val))

            next_values = {}
            V_inv = inv_vands[d + 1]
            V_inv_flat = [val for row in V_inv for val in row]
            V_inv_mat = flint.nmod_mat(d + 1, d + 1, V_inv_flat, p_val)

            for rem, vi_vals in grouped.items():
                vi_vals.sort(key=lambda x: x[0])
                y = [val for _, val in vi_vals]
                y_mat = flint.nmod_mat(d + 1, 1, y, p_val)
                c_mat = V_inv_mat * y_mat

                for j in range(d + 1):
                    new_pt = rem[:step] + (j,) + rem[step:]
                    next_values[new_pt] = int(c_mat[j, 0])
            current_values = next_values
        return current_values

    def eval_point_mod_p(pt: Tuple[int, ...], p_val: int) -> int:
        M = []
        for r in range(n):
            row = []
            for c in range(n):
                val = 0
                for pt_val, int_A in zip(pt, integer_matrices):
                    val = (val + pt_val * int_A[r][c]) % p_val
                row.append(val)
            M.append(row)
        return int(modular_det(M, p_val))

    try:
        inv_vands = get_inverse_vandermonde_matrices_cached(n + 1, p)

        if HAS_CYTHON:
            grid_pts_np = np.array([pt + (1,) for pt in full_grid_pts], dtype=np.int64)
            matrices_np = np.array(integer_matrices, dtype=np.int64)
            dets = list(eval_points_grid_mod_p(matrices_np, grid_pts_np, p))
            values = dict(zip(full_grid_pts, dets))
        else:
            values = {}
            for pt in full_grid_pts:
                values[pt] = eval_point_mod_p(pt + (1,), p)

        interpolated = interpolate_full_grid(values, m - 1, n, p, inv_vands)
        c_p = [interpolated.get(exp[:-1], 0) for exp in exps]
        return p, c_p
    except Exception:
        return p, None


class ParallelScheduler:
    def __init__(self, backend: str = "threads", max_workers: Optional[int] = None) -> None:
        self.backend = backend.lower()
        if self.backend not in ("threads", "processes", "sequential"):
            raise ValueError(f"Unknown backend: {backend}")
        self.max_workers = max_workers or os.cpu_count() or 1
        self._executor: Optional[concurrent.futures.Executor] = None

    def __enter__(self) -> "ParallelScheduler":
        if self.backend == "threads":
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        elif self.backend == "processes":
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def evaluate(self, worker_func: Callable[..., Any], items: List[Any], *args: Any) -> List[Any]:
        """
        Evaluate items using worker_func(item, *args).
        Automatically falls back to sequential execution if no executor is active,
        or if backend is set to "sequential".
        """
        if self.backend == "sequential" or self._executor is None:
            return [worker_func(item, *args) for item in items]

        futures = [self._executor.submit(worker_func, item, *args) for item in items]
        return [f.result() for f in futures]
