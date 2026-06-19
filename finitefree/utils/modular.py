import functools
from typing import Any

import flint

__all__ = [
    "modular_det",
    "crt",
    "prime_generator",
    "get_inverse_vandermonde_matrix",
    "get_inverse_vandermonde_matrices_cached",
]

try:
    import importlib

    _modular_fast = importlib.import_module(".modular_fast", package=__package__)
    modular_det = _modular_fast.modular_det
except ImportError:

    def modular_det(A: list[list[int]], p: int) -> int:
        """Computes the determinant of a matrix A modulo prime p in O(n^3) time."""
        n = len(A)
        det = 1
        M = [[A[i][j] % p for j in range(n)] for i in range(n)]
        for i in range(n):
            pivot = -1
            for r in range(i, n):
                if M[r][i] != 0:
                    pivot = r
                    break
            if pivot == -1:
                return 0
            if pivot != i:
                M[i], M[pivot] = M[pivot], M[i]
                det = (p - det) % p
            val = M[i][i]
            det = (det * val) % p
            inv = pow(val, -1, p)
            for r in range(i + 1, n):
                factor = (M[r][i] * inv) % p
                for c in range(i, n):
                    M[r][c] = (M[r][c] - factor * M[i][c]) % p
        return det


def crt(modulo_values: list[int], primes: list[int]) -> int:
    import os

    if os.environ.get("PYFFP_DISABLE_CYTHON") != "1":
        try:
            import numpy as np

            from .modular_fast import crt as crt_fast

            mix_np = np.array(modulo_values, dtype=np.int64)
            primes_np = np.array(primes, dtype=np.int64)
            return int(crt_fast(mix_np, primes_np))
        except (ImportError, AttributeError, ValueError):
            pass

    n_p = len(primes)
    mix = list(modulo_values)
    c = [1] * n_p
    for i in range(1, n_p):
        c_val = 1
        for j in range(i):
            c_val = (c_val * primes[j]) % primes[i]
        c_inv = pow(c_val, -1, primes[i])
        c[i] = c_inv
    u = [0] * n_p
    u[0] = mix[0] % primes[0]
    for i in range(1, n_p):
        val = u[0]
        p_prod = 1
        for j in range(1, i):
            p_prod = (p_prod * primes[j - 1]) % primes[i]
            val = (val + u[j] * p_prod) % primes[i]
        u[i] = ((mix[i] - val) * c[i]) % primes[i]
    x = u[0]
    p_prod = 1
    M = primes[0]
    for i in range(1, n_p):
        p_prod *= primes[i - 1]
        x += u[i] * p_prod
        M *= primes[i]
    if x > M // 2:
        x -= M
    return x


def prime_generator(start: int = 1000000007) -> Any:
    curr = start
    while True:
        if flint.fmpz(curr).is_probable_prime():
            yield curr
        curr += 1


@functools.lru_cache(maxsize=None)
def get_inverse_vandermonde_matrix(k: int, p: int) -> list[list[int]]:
    V = flint.nmod_mat(k, k, p)
    for r in range(k):
        for c in range(k):
            V[r, c] = pow(r, c, p)
    I_mat = flint.nmod_mat(k, k, p)
    for r in range(k):
        I_mat[r, r] = 1
    V_inv = V.solve(I_mat)
    return [[int(V_inv[r, c]) for c in range(k)] for r in range(k)]


def get_inverse_vandermonde_matrices_cached(
    max_k: int, p: int
) -> dict[int, list[list[int]]]:
    return {k: get_inverse_vandermonde_matrix(k, p) for k in range(1, max_k + 1)}
