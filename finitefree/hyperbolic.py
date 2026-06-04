from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class StraightLineProgram:
    """
    Efficient representation for computing gradients and Hessians of hyperbolic
    polynomials without explicit monomial enumeration.
    """

    def __init__(self, operations: Sequence[Any], pencil: Any = None) -> None:
        """
        operations: A sequence of operations representing the polynomial evaluation.
        pencil: An optional SymmetricMatrixPencil.
        """
        self.operations = operations
        self.pencil = pencil

    def evaluate(self, x: NDArray[np.float64]) -> float:
        """
        Evaluates the polynomial at the point x.
        """
        if self.pencil is None:
            raise NotImplementedError("Pencil not provided to SLP")
        A_val = self.pencil.evaluate(x)
        return float(np.linalg.det(A_val))

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the gradient of the polynomial at point x using Jacobi's formula.
        """
        if self.pencil is None:
            raise NotImplementedError("Pencil not provided to SLP")
        A_val = self.pencil.evaluate(x)
        det_A = np.linalg.det(A_val)
        if np.abs(det_A) < 1e-15:
            inv_A = np.linalg.pinv(A_val)
        else:
            inv_A = np.linalg.inv(A_val)

        grad = np.zeros(self.pencil.m, dtype=np.float64)
        for i, Ai in enumerate(self.pencil.matrices):
            grad[i] = det_A * np.trace(inv_A @ Ai)
        return grad

    def hessian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the Hessian matrix at point x using Jacobi's formula derivative.
        """
        if self.pencil is None:
            raise NotImplementedError("Pencil not provided to SLP")
        A_val = self.pencil.evaluate(x)
        det_A = np.linalg.det(A_val)
        if np.abs(det_A) < 1e-15:
            inv_A = np.linalg.pinv(A_val)
        else:
            inv_A = np.linalg.inv(A_val)

        m = self.pencil.m
        hess = np.zeros((m, m), dtype=np.float64)

        # Precompute B_i = inv_A @ A_i to avoid O(n^3) matrix multiplications
        # inside the nested loop
        B = [inv_A @ Ai for Ai in self.pencil.matrices]
        traces = [np.trace(Bi) for Bi in B]
        B_T = [Bi.T for Bi in B]

        for i in range(m):
            for j in range(m):
                term1 = traces[i] * traces[j]
                # tr(B_i @ B_j) = sum(B_i * B_j.T) which is O(n^2) instead of O(n^3)
                term2 = np.sum(B[i] * B_T[j])
                hess[i, j] = det_A * (term1 - term2)
        return hess


class SymmetricMatrixPencil:
    """
    Data structures mapping real-rooted univariate polynomials to their
    hyperbolic multivariate generalizations.
    """

    def __init__(self, matrices: Sequence[NDArray[np.float64]]) -> None:
        r"""
        matrices: A list of symmetric matrices A_1, A_2, ..., A_m
        The pencil is defined as \sum x_i A_i
        """
        self.matrices = [np.array(A, dtype=np.float64) for A in matrices]
        self.m = len(self.matrices)
        if self.m > 0:
            self.n = self.matrices[0].shape[0]
            for A in self.matrices:
                if A.shape != (self.n, self.n):
                    raise ValueError(
                        "All matrices in the pencil must have the same shape"
                    )
                if not np.allclose(A, A.T):
                    raise ValueError("All matrices must be symmetric")

    def evaluate(self, x: Sequence[float]) -> NDArray[np.float64]:
        r"""
        Evaluates the matrix pencil at point x = (x_1, ..., x_m)
        Returns the matrix \sum x_i A_i
        """
        if len(x) != self.m:
            raise ValueError(f"Expected {self.m} variables, got {len(x)}")

        result = np.zeros((self.n, self.n), dtype=np.float64)
        for xi, A in zip(x, self.matrices):
            result += xi * A
        return result

    def verify_hyperbolicity(self, e: Sequence[float]) -> bool:
        """
        Verifies if the pencil is hyperbolic in direction e using
        definite matrix programming LMI: A(e) = sum e_i A_i > 0
        """
        A_e = self.evaluate(e)
        if not np.allclose(A_e, A_e.T):
            return False
        # strictly positive eigenvalues for positive definiteness
        eigvals = np.linalg.eigvalsh(A_e)
        return bool(np.all(eigvals > 1e-14))

    def diagonal_specialization(self, w: Sequence[Any], b: Sequence[Any]) -> Any:
        """
        Computes the univariate polynomial
        P(w_1 z + b_1, ..., w_m z + b_m) = det(z A + B)
        exactly where A = sum w_i A_i and B = sum b_i A_i.
        Returns a RealRootedPolynomial.
        """
        import math

        import flint
        import sympy as sp

        from .core import RealRootedPolynomial

        if len(w) != self.m or len(b) != self.m:
            raise ValueError(
                f"Weights and bias lengths must match pencil variables ({self.m})."
            )

        exact_A_coeffs = [sp.sympify(wi) for wi in w]
        exact_b_coeffs = [sp.sympify(bi) for bi in b]

        exact_matrices = []
        for mat in self.matrices:
            row_list = []
            for r in range(self.n):
                col_list = []
                for c in range(self.n):
                    val = sp.sympify(mat[r, c])
                    if isinstance(val, (int, float, np.number)):
                        val = sp.Rational(val)
                    col_list.append(val)
                row_list.append(col_list)
            exact_matrices.append(row_list)

        A_exact = [[sp.Integer(0) for _ in range(self.n)] for _ in range(self.n)]
        B_exact = [[sp.Integer(0) for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.m):
            wi = exact_A_coeffs[i]
            bi = exact_b_coeffs[i]
            exact_mat = exact_matrices[i]
            for r in range(self.n):
                for c in range(self.n):
                    A_exact[r][c] += wi * exact_mat[r][c]
                    B_exact[r][c] += bi * exact_mat[r][c]

        denominators = []
        for r in range(self.n):
            for c in range(self.n):
                val_A = A_exact[r][c]
                val_B = B_exact[r][c]
                if isinstance(val_A, sp.Rational):
                    denominators.append(val_A.q)
                if isinstance(val_B, sp.Rational):
                    denominators.append(val_B.q)

        D = 1
        for den in denominators:
            D = (D * den) // math.gcd(D, den)

        A_int = [[int(A_exact[r][c] * D) for c in range(self.n)] for r in range(self.n)]
        B_int = [[int(B_exact[r][c] * D) for c in range(self.n)] for r in range(self.n)]

        def eval_mod_p(z_val: int, p: int) -> int:
            M_pt = flint.nmod_mat(self.n, self.n, p)
            for r in range(self.n):
                for c in range(self.n):
                    M_pt[r, c] = (z_val * A_int[r][c] + B_int[r][c]) % p
            return int(M_pt.det())

        def prime_generator(start: int = 1000000007) -> Any:
            curr = start
            while True:
                if flint.fmpz(curr).is_probable_prime():
                    yield curr
                curr += 1

        def crt(modulo_values: list[int], primes: list[int]) -> int:
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

        primes_gen = prime_generator(1000000007)
        reconstructed = None
        primes_used = []
        coeffs_by_prime = []

        deg = self.n

        while True:
            p = next(primes_gen)
            V = flint.nmod_mat(deg + 1, deg + 1, p)
            for r in range(deg + 1):
                for c in range(deg + 1):
                    V[r, c] = pow(r, c, p)
            I_mat = flint.nmod_mat(deg + 1, deg + 1, p)
            for r in range(deg + 1):
                I_mat[r, r] = 1
            try:
                V_inv = V.solve(I_mat)
            except Exception:
                continue

            y = [eval_mod_p(k, p) for k in range(deg + 1)]
            c_p = [0] * (deg + 1)
            for r in range(deg + 1):
                s = 0
                for col in range(deg + 1):
                    s = (s + int(V_inv[r, col]) * y[col]) % p
                c_p[r] = s

            coeffs_by_prime.append(c_p)
            primes_used.append(p)

            if len(primes_used) >= 2:
                current_reconstruction = []
                for i in range(deg + 1):
                    vals = [coeffs_by_prime[k][i] for k in range(len(primes_used))]
                    current_reconstruction.append(crt(vals, primes_used))

                if (
                    reconstructed is not None
                    and current_reconstruction == reconstructed
                ):
                    reconstructed = current_reconstruction
                    break
                reconstructed = current_reconstruction

        c_coeffs = []
        denom_scale = D**self.n
        for val in reconstructed:
            if val % denom_scale == 0:
                c_coeffs.append(val // denom_scale)
            else:
                c_coeffs.append(sp.Rational(val, denom_scale))

        return RealRootedPolynomial(list(reversed(c_coeffs)))

    def characteristic_polynomial_slp(self) -> StraightLineProgram:
        """
        Converts the characteristic polynomial of the symmetric matrix pencil
        into a StraightLineProgram for efficient gradient/Hessian queries.
        """
        return StraightLineProgram(operations=["det"], pencil=self)


class MultiplicativeMatrixPencil:
    """
    Data structures for generalized asymmetric matrix pencils representing
    algebraic geometries of multiplicative convolutions.
    """

    def __init__(self, matrices: Sequence[NDArray[np.float64]]) -> None:
        self.matrices = [np.array(A, dtype=np.float64) for A in matrices]
        self.m = len(self.matrices)
        if self.m > 0:
            self.n = self.matrices[0].shape[0]
            for A in self.matrices:
                if A.shape != (self.n, self.n):
                    raise ValueError(
                        "All matrices in the pencil must have the same shape"
                    )

    def evaluate(self, x: Sequence[float]) -> NDArray[np.float64]:
        if len(x) != self.m:
            raise ValueError(f"Expected {self.m} variables, got {len(x)}")
        result = np.zeros((self.n, self.n), dtype=np.float64)
        for xi, A in zip(x, self.matrices):
            result += xi * A
        return result
