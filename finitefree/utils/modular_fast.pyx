# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnpt
from libc.stdlib cimport malloc, free

cdef long long mod_inv(long long a, long long m) noexcept nogil:
    cdef long long m0 = m
    cdef long long y = 0, x = 1
    cdef long long q, t
    
    if m == 1:
        return 0
        
    while a > 1:
        q = a // m
        t = m
        m = a % m
        a = t
        t = y
        y = x - q * y
        x = t
        
    if x < 0:
        x = x + m0
        
    return x


cdef long long mod_pow(long long base, long long exp, long long mod) noexcept nogil:
    cdef long long res = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        exp = exp // 2
    return res


def construct_zippel_vandermonde_mod_p(long long[:, :] test_pts, long long[:, :] candidates, long long p):
    cdef int K = test_pts.shape[0]
    cdef int num_vars = test_pts.shape[1]
    cdef int r, c, v
    cdef long long term, val
    cdef int power
    cdef long long[:, :] V = np.zeros((K, K), dtype=np.int64)
    
    with nogil:
        for r in range(K):
            for c in range(K):
                term = 1
                for v in range(num_vars):
                    val = test_pts[r, v]
                    power = candidates[c, v]
                    term = (term * mod_pow(val, power, p)) % p
                V[r, c] = term
    return V


def modular_det(list A, long long p):
    cdef int n = len(A)
    cdef long long det = 1
    cdef int i, r, c, pivot
    cdef long long val, inv, factor
    
    # We use a 2D memoryview for fast typed indexing
    cdef long long[:, :] M = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        row = A[i]
        for c in range(n):
            M[i, c] = (<long long>row[c]) % p
            
    for i in range(n):
        pivot = -1
        for r in range(i, n):
            if M[r, i] != 0:
                pivot = r
                break
        if pivot == -1:
            return 0
        if pivot != i:
            # swap rows i and pivot
            for c in range(n):
                val = M[i, c]
                M[i, c] = M[pivot, c]
                M[pivot, c] = val
            det = (p - det) % p
            
        val = M[i, i]
        det = (det * val) % p
        
        inv = mod_inv(val, p)
        
        for r in range(i + 1, n):
            factor = (M[r, i] * inv) % p
            for c in range(i, n):
                M[r, c] = (M[r, c] - factor * M[i, c]) % p
                if M[r, c] < 0:
                    M[r, c] += p
                    
    return det


cdef void _eval_diagonal_specialization_mod_p_c(
    long long[:, :] A,
    long long[:, :] B,
    long long p,
    int deg,
    long long[:, :] temp,
    long long[:] y
) noexcept nogil:
    cdef int n = A.shape[0]
    cdef int z_val, r, c, i, pivot
    cdef long long val, inv, factor, det

    for z_val in range(deg + 1):
        for r in range(n):
            for c in range(n):
                val = (z_val * A[r, c] + B[r, c]) % p
                if val < 0:
                    val += p
                temp[r, c] = val

        # Compute determinant of temp modulo p
        det = 1
        for i in range(n):
            pivot = -1
            for r in range(i, n):
                if temp[r, i] != 0:
                    pivot = r
                    break
            if pivot == -1:
                det = 0
                break
            if pivot != i:
                for c in range(n):
                    val = temp[i, c]
                    temp[i, c] = temp[pivot, c]
                    temp[pivot, c] = val
                det = (p - det) % p

            val = temp[i, i]
            det = (det * val) % p
            inv = mod_inv(val, p)

            for r in range(i + 1, n):
                factor = (temp[r, i] * inv) % p
                for c in range(i, n):
                    temp[r, c] = (temp[r, c] - factor * temp[i, c]) % p
                    if temp[r, c] < 0:
                        temp[r, c] += p

        y[z_val] = det


def eval_diagonal_specialization_mod_p(long long[:, :] A, long long[:, :] B, long long p, int deg):
    cdef int n = A.shape[0]
    cdef long long[:, :] temp = np.zeros((n, n), dtype=np.int64)
    cdef long long[:] y = np.zeros(deg + 1, dtype=np.int64)
    with nogil:
        _eval_diagonal_specialization_mod_p_c(A, B, p, deg, temp, y)
    return y


cdef void _eval_points_grid_mod_p_c(
    long long[:, :, :] matrices,
    long long[:, :] grid_pts,
    long long p,
    long long[:, :] temp,
    long long[:] results
) noexcept nogil:
    cdef int num_pts = grid_pts.shape[0]
    cdef int m = grid_pts.shape[1]
    cdef int n = matrices.shape[1]
    cdef int i, r, c, j, pivot, row_i
    cdef long long val, inv, factor, det

    for i in range(num_pts):
        for r in range(n):
            for c in range(n):
                val = 0
                for j in range(m):
                    val = (val + grid_pts[i, j] * matrices[j, r, c]) % p
                val = val % p
                if val < 0:
                    val += p
                temp[r, c] = val

        # Compute determinant of temp modulo p
        det = 1
        for row_i in range(n):
            pivot = -1
            for r in range(row_i, n):
                if temp[r, row_i] != 0:
                    pivot = r
                    break
            if pivot == -1:
                det = 0
                break
            if pivot != row_i:
                for c in range(n):
                    val = temp[row_i, c]
                    temp[row_i, c] = temp[pivot, c]
                    temp[pivot, c] = val
                det = (p - det) % p

            val = temp[row_i, row_i]
            det = (det * val) % p
            inv = mod_inv(val, p)

            for r in range(row_i + 1, n):
                factor = (temp[r, row_i] * inv) % p
                for c in range(row_i, n):
                    temp[r, c] = (temp[r, c] - factor * temp[row_i, c]) % p
                    if temp[r, c] < 0:
                        temp[r, c] += p

        results[i] = det


def eval_points_grid_mod_p(long long[:, :, :] matrices, long long[:, :] grid_pts, long long p):
    cdef int num_pts = grid_pts.shape[0]
    cdef int n = matrices.shape[1]
    cdef long long[:, :] temp = np.zeros((n, n), dtype=np.int64)
    cdef long long[:] results = np.zeros(num_pts, dtype=np.int64)
    with nogil:
        _eval_points_grid_mod_p_c(matrices, grid_pts, p, temp, results)
    return results


def crt(long long[::1] mix, long long[::1] primes):

    cdef int n_p = primes.shape[0]
    if n_p == 0:
        return 0

    cdef long long* c = <long long*>malloc(n_p * sizeof(long long))
    cdef long long* u = <long long*>malloc(n_p * sizeof(long long))
    if not c or not u:
        if c:
            free(c)
        if u:
            free(u)
        raise MemoryError()

    cdef int i, j
    cdef long long c_val, c_inv, val, p_prod_mod
    cdef object x, p_prod, M

    try:
        c[0] = 1
        for i in range(1, n_p):
            c_val = 1
            for j in range(i):
                c_val = (c_val * primes[j]) % primes[i]
            c_inv = mod_inv(c_val, primes[i])
            c[i] = c_inv

        u[0] = mix[0] % primes[0]
        if u[0] < 0:
            u[0] += primes[0]

        for i in range(1, n_p):
            val = u[0]
            p_prod_mod = 1
            for j in range(1, i):
                p_prod_mod = (p_prod_mod * primes[j - 1]) % primes[i]
                val = (val + u[j] * p_prod_mod) % primes[i]

            c_val = (mix[i] - val) % primes[i]
            if c_val < 0:
                c_val += primes[i]
            u[i] = (c_val * c[i]) % primes[i]

        x = u[0]
        p_prod = 1
        M = primes[0]

        for i in range(1, n_p):
            p_prod = p_prod * primes[i - 1]
            x = x + u[i] * p_prod
            M = M * primes[i]

        if x > M // 2:
            x = x - M

        return x
    finally:
        free(c)
        free(u)
