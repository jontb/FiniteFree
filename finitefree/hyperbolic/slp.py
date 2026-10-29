import operator
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.conversion import flint_to_float, sympy_to_fmpq

_BINARY_OPERATIONS = frozenset({"add", "+", "sub", "-", "mul", "*", "div", "/"})
_UNARY_OPERATIONS = frozenset({"neg"})


def _operand_index(value: Any, position: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Operation {position} has a boolean operand index")
    try:
        index = operator.index(value)
    except TypeError as error:
        raise ValueError(
            f"Operation {position} operand indices must be integers"
        ) from error
    if index < 0:
        raise ValueError(f"Operation {position} operand indices must be nonnegative")
    return index


def _normalize_operation(operation: Any, position: int) -> tuple[Any, ...]:
    if isinstance(operation, (str, bytes)) or not isinstance(operation, (tuple, list)):
        raise ValueError(
            f"Operation {position} must be a tuple (code, operands...), "
            f"got {operation!r}"
        )
    if len(operation) == 0:
        raise ValueError(f"Operation {position} is empty")
    code = operation[0]
    if isinstance(code, str) and code == "const":
        if len(operation) != 2:
            raise ValueError(f"Operation {position} 'const' takes exactly one value")
        return ("const", operation[1])
    if isinstance(code, str) and code in _BINARY_OPERATIONS:
        arity = 2
    elif isinstance(code, str) and code in _UNARY_OPERATIONS:
        arity = 1
    else:
        raise ValueError(f"Operation {position} has unknown code {code!r}")
    if len(operation) != arity + 1:
        raise ValueError(
            f"Operation {position} {code!r} takes exactly {arity} operand(s)"
        )
    return (code,) + tuple(_operand_index(value, position) for value in operation[1:])


def _scale_by_determinant(
    determinant: float, sign: float, logdet: float, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return det(A) * values without letting det(A) alone underflow/overflow."""
    if np.isfinite(determinant) and abs(determinant) >= np.finfo(np.float64).tiny:
        return np.asarray(determinant * values, dtype=np.float64)
    with np.errstate(divide="ignore"):
        logs = logdet + np.log(np.abs(values))
    return np.asarray(sign * np.sign(values) * np.exp(logs), dtype=np.float64)


class StraightLineProgram:
    r"""
    Evaluate a scalar operation sequence or a determinant matrix pencil.

    ``operations == ["det"]`` (any sequence holding only ``"det"``) selects the
    determinant of ``pencil``. Otherwise each operation is a tuple
    ``(code, operands...)``: ``("const", value)``, binary ``add``/``+``,
    ``sub``/``-``, ``mul``/``*``, ``div``/``/`` and unary ``neg``. Operand
    indices are nonnegative positions in the value list, which starts with the
    input coordinates; operation ``k`` appends value ``len(x) + k``. Operations
    are copied and validated at construction; unknown codes, wrong arity and
    invalid operand indices raise ValueError. The result is the last value.

    Generic scalar gradients use reverse-mode differentiation; generic Hessians
    are not implemented. The determinant program uses matrix determinant/inverse
    trace identities for gradients and Hessians. Exact=True uses rational FLINT
    operations; numerical methods use float64. Exact and numerical determinant
    derivatives both reject singular evaluated matrices.
    """

    def __init__(self, operations: Sequence[Any], pencil: Any = None) -> None:
        """
        operations: ``["det"]`` or a sequence of scalar operation tuples.
        pencil: An optional SymmetricMatrixPencil.
        """
        if isinstance(operations, (str, bytes)):
            raise ValueError(
                "operations must be a sequence of operations, such as ['det']"
            )
        operation_list = list(operations)
        self._is_determinant = operation_list == ["det"]
        if self._is_determinant:
            self.operations: list[Any] = ["det"]
            self._required_inputs = 0
        else:
            self.operations = [
                _normalize_operation(operation, position)
                for position, operation in enumerate(operation_list)
            ]
            # Operation k can read values 0 .. len(x) + k - 1.
            self._required_inputs = max(
                (
                    index - position + 1
                    for position, operation in enumerate(self.operations)
                    if operation[0] != "const"
                    for index in operation[1:]
                ),
                default=0,
            )
        self.pencil = pencil

    def _require_pencil(self) -> None:
        if self.pencil is None:
            raise NotImplementedError("Pencil not provided to SLP")

    def _forward(self, x: Sequence[Any], exact: bool) -> list[Any]:
        if len(x) < self._required_inputs:
            raise ValueError(
                f"Operations reference {self._required_inputs} input coordinates, "
                f"got {len(x)}"
            )
        v: list[Any]
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
            else:  # "neg"; construction rejects every other code
                v.append(-v[op_info[1]])
        return v

    def evaluate(self, x: NDArray[np.float64], exact: bool = False) -> Any:
        """
        Evaluates the polynomial at the point x.
        """
        if self._is_determinant:
            self._require_pencil()
            if exact:
                A_exact = self._evaluate_exact(x)  # type: ignore[arg-type]
                return A_exact.det()
            A_val = self.pencil.evaluate(x)
            return float(np.linalg.det(A_val))

        return self._forward(x, exact)[-1]  # type: ignore[arg-type]

    def _evaluate_exact(self, x: Sequence[Any]) -> Any:
        return self.pencil._evaluate_exact(x)

    def _float_determinant_solves(
        self, x: NDArray[np.float64]
    ) -> tuple[tuple[float, float, float], list[NDArray[np.float64]]]:
        """Return (det, sign, log|det|) and ``A(x)^{-1} A_i`` from LU solves.

        Raises ``numpy.linalg.LinAlgError`` when LU factorization
        of ``A(x)`` meets an exactly zero pivot. No pseudoinverse or
        determinant-magnitude threshold is used, so tiny nonsingular determinants
        are differentiated rather than truncated.
        """
        A_val = self.pencil.evaluate(x)
        determinant = float(np.linalg.det(A_val))
        sign, logdet = np.sign(determinant), 0.0
        if not (
            np.isfinite(determinant) and abs(determinant) >= np.finfo(np.float64).tiny
        ):
            # Zero, subnormal or overflowing det: separate underflow from an
            # exactly zero LU pivot, and keep the log scale for the products.
            sign, logdet = np.linalg.slogdet(A_val)
        if sign == 0:
            raise np.linalg.LinAlgError(
                "Numerical determinant derivatives require a nonsingular evaluated "
                "matrix; use MultivariatePolynomial.gradient_float64/hessian_float64 "
                "or exact polynomial derivatives at singular points"
            )
        n = self.pencil.n
        stacked = np.concatenate(list(self.pencil.matrices), axis=1)
        solved = np.linalg.solve(A_val, stacked)
        blocks = [solved[:, i * n : (i + 1) * n] for i in range(self.pencil.m)]
        return (determinant, float(sign), float(logdet)), blocks

    def gradient(self, x: NDArray[np.float64], exact: bool = False) -> NDArray[Any]:
        r"""
        Compute a generic reverse-mode gradient or a determinant-pencil gradient.
        Exact determinant gradients require a nonsingular evaluated matrix and raise
        ValueError at singular points. Numerical determinant gradients solve with
        one LU factorization of the evaluated matrix and raise
        numpy.linalg.LinAlgError at an exactly zero LU pivot; they
        use no pseudoinverse or determinant threshold. Near-singular results remain
        conditioning-dependent.
        """
        if self._is_determinant:
            self._require_pencil()
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
                for Ai in self.pencil._get_matrices_exact():
                    tr = flint.fmpq(0)
                    for r in range(self.pencil.n):
                        for c in range(self.pencil.n):
                            tr += inv_A[r, c] * sympy_to_fmpq(Ai[c, r])
                    grad.append(det_A * tr)
                return np.array(grad, dtype=object)

            scale, blocks = self._float_determinant_solves(x)
            traces = np.array([np.trace(Bi) for Bi in blocks], dtype=np.float64)
            return _scale_by_determinant(*scale, traces)

        # Generic Reverse-Mode Automatic Differentiation
        m = len(x)
        one: Any
        zero: Any
        if exact:
            import flint

            one = flint.fmpq(1)
            zero = flint.fmpq(0)
        else:
            one = 1.0
            zero = 0.0

        # Forward pass to build trace
        v = self._forward(x, exact)  # type: ignore[arg-type]

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
        r"""
        Compute determinant-pencil Hessians using Jacobi's trace identities.
        A determinant program with a pencil is required; generic scalar-operation
        Hessians are not implemented and raise NotImplementedError.
        Exact=True raises ValueError at singular matrices. Numerical evaluation
        raises numpy.linalg.LinAlgError at an exactly zero LU pivot
        and remains conditioning-dependent near singular points.
        """
        if not self._is_determinant:
            raise NotImplementedError(
                "Hessians of generic straight-line programs are not implemented; "
                "only the determinant program ['det'] supports hessian()"
            )
        self._require_pencil()
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
            for Ai in self.pencil._get_matrices_exact():
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

        (determinant, sign, logdet), B_float = self._float_determinant_solves(x)
        normal = (
            np.isfinite(determinant) and abs(determinant) >= np.finfo(np.float64).tiny
        )
        # B_i = inv(A) @ A_i comes from one LU solve against all coefficients.
        if normal:
            with np.errstate(over="ignore", invalid="ignore"):
                hess = determinant * self._trace_form(B_float)
            if np.all(np.isfinite(hess)):
                return hess
        # det * (tr B_i tr B_j - tr(B_i B_j)) is quadratic in B: scaling each
        # B_i by sqrt|det| keeps tiny determinants with huge traces (or the
        # reverse) from overflowing/underflowing in the separate factors.
        root = np.sqrt(abs(determinant)) if normal else np.exp(0.5 * logdet)
        return sign * self._trace_form([root * Bi for Bi in B_float])

    def _trace_form(self, B: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Return the matrix ``tr(B_i) tr(B_j) - tr(B_i B_j)``."""
        m = len(B)
        result = np.zeros((m, m), dtype=np.float64)
        traces = [np.trace(Bi) for Bi in B]
        B_T = [Bi.T for Bi in B]
        for i in range(m):
            for j in range(m):
                term1 = traces[i] * traces[j]
                # tr(B_i @ B_j) = sum(B_i * B_j.T) which is O(n^2) instead of O(n^3)
                term2 = np.sum(B[i] * B_T[j])
                result[i, j] = term1 - term2
        return result
