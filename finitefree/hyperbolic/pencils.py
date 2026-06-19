import math
from typing import Any, List, Sequence

import flint
import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ..core import RealRootedPolynomial
from ..utils.conversion import sympy_to_fmpq
from ..utils.modular import crt, prime_generator
from ..utils.parallel import (
    ParallelScheduler,
    _eval_diagonal_specialization_prime_worker,
)


class SymmetricMatrixPencil:
    """
    Data structures mapping real-rooted univariate polynomials to their
    hyperbolic multivariate generalizations.
    """

    def __init__(self, matrices: Sequence[NDArray[np.float64]]) -> None:
        r"""
        matrices: A list of symmetric matrices $A_1, A_2, \dots, A_m$
        The pencil is defined as $\sum x_i A_i$
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
        Evaluates the matrix pencil at point $x = (x_1, \dots, x_m)$
        Returns the matrix $\sum x_i A_i$
        """
        if len(x) != self.m:
            raise ValueError(f"Expected {self.m} variables, got {len(x)}")

        result = np.zeros((self.n, self.n), dtype=np.float64)
        for xi, A in zip(x, self.matrices):
            result += xi * A
        return result

    def _get_matrices_exact(self) -> List[Any]:
        if not hasattr(self, "_matrices_exact"):
            import flint

            from ..utils.conversion import sympy_to_fmpq

            self._matrices_exact = []
            for A in self.matrices:
                mat = flint.fmpq_mat(self.n, self.n)
                for r in range(self.n):
                    for c in range(self.n):
                        mat[r, c] = sympy_to_fmpq(A[r, c])
                self._matrices_exact.append(mat)
        return self._matrices_exact

    def _get_matrices_sympy(self) -> List[List[List[Any]]]:
        if not hasattr(self, "_matrices_sympy"):
            import sympy as sp

            self._matrices_sympy = []
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
                self._matrices_sympy.append(row_list)
        return self._matrices_sympy

    def _evaluate_exact(self, x: Sequence[Any]) -> Any:

        A_exact = flint.fmpq_mat(self.n, self.n)
        exact_mats = self._get_matrices_exact()
        for xi, Ai in zip(x, exact_mats):
            A_exact += Ai * sympy_to_fmpq(xi)
        return A_exact

    def verify_hyperbolicity(self, e: Sequence[float], exact: bool = False) -> bool:
        r"""
        Verifies if the pencil is hyperbolic in direction $e$ using
        definite matrix programming LMI: $A(e) = \sum e_i A_i \succ 0$
        """
        if exact:
            A_exact = self._evaluate_exact(e)

            # Check symmetry
            for r in range(self.n):
                for c in range(r + 1, self.n):
                    if A_exact[r, c] != A_exact[c, r]:
                        return False

            p_poly = A_exact.charpoly()
            p_roots = RealRootedPolynomial(p_poly, assume_real_rooted=True)
            return p_roots.has_strictly_positive_roots

        A_e = self.evaluate(e)
        if not np.allclose(A_e, A_e.T):
            return False
        # strictly positive eigenvalues for positive definiteness
        eigvals = np.linalg.eigvalsh(A_e)
        return bool(np.all(eigvals > 1e-14))

    def diagonal_specialization(
        self,
        w: Sequence[Any],
        b: Sequence[Any],
        parallel: bool = False,
        backend: str = "threads",
    ) -> Any:
        r"""
        Computes the univariate polynomial
        $P(w_1 z + b_1, \dots, w_m z + b_m) = \det(z A + B)$
        exactly where $A = \sum w_i A_i$ and $B = \sum b_i A_i$.
        Returns a RealRootedPolynomial.
        """

        if len(w) != self.m or len(b) != self.m:
            raise ValueError(
                f"Weights and bias lengths must match pencil variables ({self.m})."
            )

        exact_A_coeffs = [sp.sympify(wi) for wi in w]
        exact_b_coeffs = [sp.sympify(bi) for bi in b]

        exact_matrices = self._get_matrices_sympy()

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

        primes_gen = prime_generator(1000000007)
        reconstructed = None
        primes_used = []
        coeffs_by_prime = []

        deg = self.n

        effective_backend = backend if parallel else "sequential"
        if self.n <= 3:
            effective_backend = "sequential"

        with ParallelScheduler(backend=effective_backend) as scheduler:
            batch_size = max(4, scheduler.max_workers)
            while True:
                batch_primes = [next(primes_gen) for _ in range(batch_size)]
                results = scheduler.evaluate(
                    _eval_diagonal_specialization_prime_worker,
                    batch_primes,
                    A_int,
                    B_int,
                    self.n,
                    deg,
                )
                for p_res, c_p in results:
                    if c_p is not None:
                        coeffs_by_prime.append(c_p)
                        primes_used.append(p_res)

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

    def characteristic_polynomial_slp(self) -> Any:
        """
        Converts the characteristic polynomial of the symmetric matrix pencil
        into a StraightLineProgram for efficient gradient/Hessian queries.
        """
        from .slp import StraightLineProgram

        return StraightLineProgram(operations=["det"], pencil=self)

    def characteristic_polynomial(self, x: Sequence[Any]) -> flint.fmpq_poly:
        r"""
        Computes the characteristic polynomial $\det(t I - A(x))$ of the pencil evaluated at $x$
        exactly using rational interpolation over $\mathbb{Q}[t]$.
        """
        import flint

        A_exact = self._evaluate_exact(x)
        n = self.n

        xs = [flint.fmpq(j) for j in range(n + 1)]
        ys = []
        for j in range(n + 1):
            tj = xs[j]
            M = flint.fmpq_mat(n, n)
            for r in range(n):
                for c in range(n):
                    if r == c:
                        M[r, c] = tj - A_exact[r, c]
                    else:
                        M[r, c] = -A_exact[r, c]
            ys.append(M.det())

        # Exact Lagrange interpolation
        t_var = flint.fmpq_poly([0, 1])
        poly_sum = flint.fmpq_poly([0])
        for i in range(n + 1):
            denom = flint.fmpq(1)
            L_i = flint.fmpq_poly([1])
            for j in range(n + 1):
                if j != i:
                    L_i *= t_var - xs[j]
                    denom *= xs[i] - xs[j]
            poly_sum += L_i * (ys[i] / denom)

        return poly_sum


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

    def _get_matrices_exact(self) -> List[Any]:
        if not hasattr(self, "_matrices_exact"):
            import flint

            from ..utils.conversion import sympy_to_fmpq

            self._matrices_exact = []
            for A in self.matrices:
                mat = flint.fmpq_mat(self.n, self.n)
                for r in range(self.n):
                    for c in range(self.n):
                        mat[r, c] = sympy_to_fmpq(A[r, c])
                self._matrices_exact.append(mat)
        return self._matrices_exact

    def _evaluate_exact(self, x: Sequence[Any]) -> Any:
        import flint

        from ..utils.conversion import sympy_to_fmpq

        A_exact = flint.fmpq_mat(self.n, self.n)
        exact_mats = self._get_matrices_exact()
        for xi, Ai in zip(x, exact_mats):
            A_exact += Ai * sympy_to_fmpq(xi)
        return A_exact

    def characteristic_polynomial(self, x: Sequence[Any]) -> flint.fmpq_poly:
        r"""
        Computes the characteristic polynomial $\det(t I - A(x))$ of the pencil evaluated at $x$
        exactly using rational interpolation over $\mathbb{Q}[t]$.
        """
        import flint

        A_exact = self._evaluate_exact(x)
        n = self.n

        xs = [flint.fmpq(j) for j in range(n + 1)]
        ys = []
        for j in range(n + 1):
            tj = xs[j]
            M = flint.fmpq_mat(n, n)
            for r in range(n):
                for c in range(n):
                    if r == c:
                        M[r, c] = tj - A_exact[r, c]
                    else:
                        M[r, c] = -A_exact[r, c]
            ys.append(M.det())

        # Exact Lagrange interpolation
        t_var = flint.fmpq_poly([0, 1])
        poly_sum = flint.fmpq_poly([0])
        for i in range(n + 1):
            denom = flint.fmpq(1)
            L_i = flint.fmpq_poly([1])
            for j in range(n + 1):
                if j != i:
                    L_i *= t_var - xs[j]
                    denom *= xs[i] - xs[j]
            poly_sum += L_i * (ys[i] / denom)

        return poly_sum

    def _get_matrices_sympy(self) -> List[List[List[Any]]]:
        if not hasattr(self, "_matrices_sympy"):
            import sympy as sp

            self._matrices_sympy = []
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
                self._matrices_sympy.append(row_list)
        return self._matrices_sympy

    def characteristic_polynomial_slp(self) -> Any:
        """
        Converts the characteristic polynomial of the multiplicative matrix pencil
        into a StraightLineProgram for efficient gradient/Hessian queries.
        """
        from .slp import StraightLineProgram

        return StraightLineProgram(operations=["det"], pencil=self)
