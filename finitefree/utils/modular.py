import functools
import math
import os
from typing import Any, Sequence

import flint
import numpy as np
from numpy.typing import NDArray


def _determinant_coefficient_bound(
    matrices: Sequence[Sequence[Sequence[int]]],
) -> int:
    """Bound every coefficient of det(sum_j x_j A_j) for integer matrices.

    Each determinant permutation contributes a product of linear forms.
    The coefficient l1 norm of that product is at most the product of their
    coefficient l1 norms. Summing over n! permutations gives the rowwise bound
    n! * product_r max_c sum_j abs(A_j[r,c]).
    """
    n = len(matrices[0])
    return math.factorial(n) * math.prod(
        max(
            (sum(abs(matrix[r][c]) for matrix in matrices) for c in range(n)),
            default=0,
        )
        for r in range(n)
    )


def _as_modular_array(
    values: Any, p: int, shape: tuple[int, ...] | None = None
) -> NDArray[np.int64]:
    """Reduce Python integers before narrowing them for Cython arithmetic.

    Nested lists cannot record trailing zero dimensions: an empty ``(m, 0, 0)``
    stack becomes shape ``(m, 0)``. Pass ``shape`` to restore it exactly.
    """
    reduced = np.asarray(np.asarray(values, dtype=object) % p, dtype=np.int64)
    if shape is not None:
        reduced = reduced.reshape(shape)
    return reduced


# Dense modular reconstruction has no expected unlucky primes: Vandermonde
# nodes 0..n are distinct modulo every prime used. Bound retries anyway, as the
# sparse path does, so a persistent per-prime failure surfaces as an error.
_MAX_CONSECUTIVE_FAILED_PRIMES = 8


__all__ = [
    "modular_det",
    "crt",
    "prime_generator",
    "get_inverse_vandermonde_matrix",
    "get_inverse_vandermonde_matrices_cached",
]

# Largest modulus m = isqrt(2**63 - 1): residue products and sums stay in int64.
_INT64_SAFE_MODULUS = math.isqrt(2**63 - 1)


def _compiled_modulus(p: Any) -> bool:
    """Whether p fits the int64 residue arithmetic of the compiled kernels."""
    try:
        return bool(2 <= p <= _INT64_SAFE_MODULUS)
    except TypeError:
        return False


def _modular_det_python(A: Sequence[Sequence[Any]], p: int) -> int:
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


_modular_det_fast: Any = None
if os.environ.get("PYFFP_DISABLE_CYTHON") != "1":
    try:
        import importlib

        _modular_det_fast = importlib.import_module(
            ".modular_fast", package=__package__
        ).modular_det
    except ImportError:
        pass


def modular_det(A: Sequence[Sequence[Any]], p: int) -> int:
    """Determinant of the integer matrix A modulo prime p, in [0, p).

    Entries may be negative or exceed int64; both implementations reduce them
    with Python integer arithmetic. The compiled kernel handles list inputs and
    primes up to isqrt(2**63 - 1) unless PYFFP_DISABLE_CYTHON=1 was set at
    import; other inputs use the Python implementation.
    """
    if _modular_det_fast is not None and isinstance(A, list) and _compiled_modulus(p):
        return int(_modular_det_fast(A, p))
    return _modular_det_python(A, p)


def crt(modulo_values: list[int], primes: list[int]) -> int:
    """Return x with x = modulo_values[i] (mod primes[i]) and |x| <= M // 2.

    Residues may be any integers. The compiled kernel is used when every prime
    is at most isqrt(2**63 - 1) unless PYFFP_DISABLE_CYTHON=1; larger primes
    use exact Python integer arithmetic.
    """
    if (
        os.environ.get("PYFFP_DISABLE_CYTHON") != "1"
        and len(modulo_values) == len(primes)
        and all(_compiled_modulus(p) for p in primes)
    ):
        try:
            from .modular_fast import crt as crt_fast

            # Reduce with Python integers before narrowing to int64.
            mix_np = np.array(
                [int(v) % int(p) for v, p in zip(modulo_values, primes)],
                dtype=np.int64,
            )
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
