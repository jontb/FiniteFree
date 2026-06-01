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

    def characteristic_polynomial_slp(self) -> StraightLineProgram:
        """
        Converts the characteristic polynomial of the symmetric matrix pencil
        into a StraightLineProgram for efficient gradient/Hessian queries.
        """
        return StraightLineProgram(operations=["det"], pencil=self)
