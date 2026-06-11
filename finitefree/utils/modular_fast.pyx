# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnpt

cdef long mod_inv(long a, long m) noexcept nogil:
    cdef long m0 = m
    cdef long y = 0, x = 1
    cdef long q, t
    
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


cdef long mod_pow(long base, long exp, long mod) noexcept nogil:
    cdef long res = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        exp = exp // 2
    return res


def construct_zippel_vandermonde_mod_p(long[:, :] test_pts, long[:, :] candidates, long p):
    cdef int K = test_pts.shape[0]
    cdef int num_vars = test_pts.shape[1]
    cdef int r, c, v
    cdef long term, val
    cdef int power
    cdef long[:, :] V = np.zeros((K, K), dtype=np.int64)
    
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


def modular_det(list A, long p):
    cdef int n = len(A)
    cdef long det = 1
    cdef int i, r, c, pivot
    cdef long val, inv, factor
    
    # We use a 2D memoryview for fast typed indexing
    cdef long[:, :] M = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        row = A[i]
        for c in range(n):
            M[i, c] = (<long>row[c]) % p
            
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


def eval_diagonal_specialization_mod_p(long[:, :] A, long[:, :] B, long p, int deg):
    cdef int n = A.shape[0]
    cdef int z_val, r, c, i, pivot
    cdef long val, inv, factor, det
    cdef long[:, :] temp = np.zeros((n, n), dtype=np.int64)
    cdef long[:] y = np.zeros(deg + 1, dtype=np.int64)
    
    with nogil:
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
            
    return y


def eval_points_grid_mod_p(long[:, :, :] matrices, long[:, :] grid_pts, long p):
    cdef int num_pts = grid_pts.shape[0]
    cdef int m = grid_pts.shape[1]
    cdef int n = matrices.shape[1]
    cdef int i, r, c, j, pivot, row_i
    cdef long val, inv, factor, det
    cdef long[:, :] temp = np.zeros((n, n), dtype=np.int64)
    cdef long[:] results = np.zeros(num_pts, dtype=np.int64)
    
    with nogil:
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
    return results
