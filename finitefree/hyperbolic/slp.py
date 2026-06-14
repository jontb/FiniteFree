from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.conversion import flint_to_float, sympy_to_fmpq


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

    def evaluate(self, x: NDArray[np.float64], exact: bool = False) -> Any:
        """
        Evaluates the polynomial at the point x.
        """
        if self.operations == ["det"]:
            if self.pencil is None:
                raise NotImplementedError("Pencil not provided to SLP")
            if exact:
                A_exact = self._evaluate_exact(x)  # type: ignore[arg-type]
                return A_exact.det()
            A_val = self.pencil.evaluate(x)
            return float(np.linalg.det(A_val))

        # Generic SLP evaluation
        if exact:
            v = [sympy_to_fmpq(xi) for xi in x]
        else:
            v = [float(xi) for xi in x]

        for op_info in self.operations:
            op = op_info[0]
            if op == "const":
                val = op_info[1]
                v.append(sympy_to_fmpq(val) if exact else float(val))
            elif op in ("add", "+"):
                v.append(v[op_info[1]] + v[op_info[2]])
            elif op in ("sub", "-"):
                v.append(v[op_info[1]] - v[op_info[2]])
            elif op in ("mul", "*"):
                v.append(v[op_info[1]] * v[op_info[2]])
            elif op in ("div", "/"):
                v.append(v[op_info[1]] / v[op_info[2]])
            elif op == "neg":
                v.append(-v[op_info[1]])

        return v[-1]

    def _evaluate_exact(self, x: Sequence[Any]) -> Any:
        return self.pencil._evaluate_exact(x)

    def gradient(self, x: NDArray[np.float64], exact: bool = False) -> NDArray[Any]:
        """
        Computes the gradient of the polynomial at point x.
        """
        if self.operations == ["det"]:
            if self.pencil is None:
                raise NotImplementedError("Pencil not provided to SLP")
            if exact:
                import flint

                A_exact = self._evaluate_exact(x)  # type: ignore[arg-type]
                det_A = A_exact.det()
                if det_A == 0:
                    raise ValueError(
                        "Exact gradient at singular point is not supported."
                    )
                identity = flint.fmpq_mat(self.pencil.n, self.pencil.n)
                for i in range(self.pencil.n):
                    identity[i, i] = flint.fmpq(1, 1)
                inv_A = A_exact.solve(identity)

                grad = []
                for Ai in self.pencil.matrices:
                    tr = flint.fmpq(0)
                    for r in range(self.pencil.n):
                        for c in range(self.pencil.n):
                            tr += inv_A[r, c] * sympy_to_fmpq(Ai[c, r])
                    grad.append(det_A * tr)
                return np.array(grad, dtype=object)

            A_val = self.pencil.evaluate(x)
            det_A = np.linalg.det(A_val)
            if np.abs(det_A) < 1e-15:
                inv_A = np.linalg.pinv(A_val)
            else:
                inv_A = np.linalg.inv(A_val)

            grad_num = np.zeros(self.pencil.m, dtype=np.float64)
            for i, Ai in enumerate(self.pencil.matrices):
                grad_num[i] = det_A * np.trace(inv_A @ Ai)
            return grad_num

        # Generic Reverse-Mode Automatic Differentiation
        m = len(x)
        one: Any
        zero: Any
        if exact:
            import flint

            v = [sympy_to_fmpq(xi) for xi in x]
            one = flint.fmpq(1)
            zero = flint.fmpq(0)
        else:
            v = [float(xi) for xi in x]
            one = 1.0
            zero = 0.0

        # Forward pass to build trace
        for op_info in self.operations:
            op = op_info[0]
            if op == "const":
                val = op_info[1]
                v.append(sympy_to_fmpq(val) if exact else float(val))
            elif op in ("add", "+"):
                v.append(v[op_info[1]] + v[op_info[2]])
            elif op in ("sub", "-"):
                v.append(v[op_info[1]] - v[op_info[2]])
            elif op in ("mul", "*"):
                v.append(v[op_info[1]] * v[op_info[2]])
            elif op in ("div", "/"):
                v.append(v[op_info[1]] / v[op_info[2]])
            elif op == "neg":
                v.append(-v[op_info[1]])

        # Backward pass
        n_total = len(v)
        adj = [zero] * n_total
        adj[-1] = one

        for k in range(n_total - 1, m - 1, -1):
            op_info = self.operations[k - m]
            op = op_info[0]
            a = adj[k]
            if op in ("add", "+"):
                adj[op_info[1]] += a
                adj[op_info[2]] += a
            elif op in ("sub", "-"):
                adj[op_info[1]] += a
                adj[op_info[2]] -= a
            elif op in ("mul", "*"):
                adj[op_info[1]] += a * v[op_info[2]]
                adj[op_info[2]] += a * v[op_info[1]]
            elif op in ("div", "/"):
                adj[op_info[1]] += a / v[op_info[2]]
                adj[op_info[2]] -= a * v[op_info[1]] / (v[op_info[2]] * v[op_info[2]])
            elif op == "neg":
                adj[op_info[1]] -= a

        if exact:
            return np.array(adj[:m], dtype=object)
        return np.array([flint_to_float(val) for val in adj[:m]], dtype=np.float64)

    def hessian(self, x: NDArray[np.float64], exact: bool = False) -> NDArray[Any]:
        """
        Computes the Hessian matrix at point x using Jacobi's formula derivative.
        """
        if self.pencil is None:
            raise NotImplementedError("Pencil not provided to SLP")
        if exact:
            import flint

            A_exact = self._evaluate_exact(x)  # type: ignore[arg-type]
            det_A = A_exact.det()
            if det_A == 0:
                raise ValueError("Exact Hessian at singular point is not supported.")
            identity = flint.fmpq_mat(self.pencil.n, self.pencil.n)
            for i in range(self.pencil.n):
                identity[i, i] = flint.fmpq(1, 1)
            inv_A = A_exact.solve(identity)

            m = self.pencil.m
            hess = np.zeros((m, m), dtype=object)
            B = []
            for Ai in self.pencil.matrices:
                Bi = flint.fmpq_mat(self.pencil.n, self.pencil.n)
                for r in range(self.pencil.n):
                    for c in range(self.pencil.n):
                        val = flint.fmpq(0)
                        for k in range(self.pencil.n):
                            val += inv_A[r, k] * sympy_to_fmpq(Ai[k, c])
                        Bi[r, c] = val
                B.append(Bi)

            traces = []
            for Bi in B:
                tr = flint.fmpq(0)
                for r in range(self.pencil.n):
                    tr += Bi[r, r]
                traces.append(tr)

            for i in range(m):
                for j in range(m):
                    term1 = traces[i] * traces[j]
                    term2 = flint.fmpq(0)
                    for r in range(self.pencil.n):
                        for c in range(self.pencil.n):
                            term2 += B[i][r, c] * B[j][c, r]
                    hess[i, j] = det_A * (term1 - term2)
            return hess

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
