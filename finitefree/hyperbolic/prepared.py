"""Prepared float64 matrix pencils; no polynomial expansion or implicit transfers."""

from __future__ import annotations

import importlib
import operator
import warnings
from contextlib import nullcontext
from typing import Any

import numpy as np
import scipy.linalg as sla


def _positive_size(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    size = operator.index(value)
    if size < 1:
        raise ValueError(f"{name} must be a positive integer")
    return size


class PreparedMatrixPencil:
    r"""Own a numerical snapshot of ``A(x) = sum_i x_i A_i``.

    ``matrices`` must be a resident float64 ndarray of shape ``(m, n, n)``
    with m,n > 0. NumPy is the default; ``backend='cupy'`` requires an installed
    CuPy and a working GPU. There is no automatic backend choice, transfer,
    dtype coercion, symmetry assumption, or fallback. Preparation copies the
    coefficients and packs solve right-hand sides once. Exact pencil methods
    remain separate; ``pencil.prepare_numeric()`` snapshots its numerical view.

    Inputs to evaluate/factor have shape ``(..., m)``, including empty batches
    and strided arrays. Outputs preserve the leading shape and backend. Finite
    inputs are validated once per call; overflow in assembly/factorization is
    rejected. All operations stay on the preparation device and current stream;
    cross-device arrays/current-device changes are rejected. GPU validation and
    solver error checks can synchronize. Callers must order concurrent streams.

    Point blocks bound assembly scratch; coefficient blocks bound derivative
    solve scratch. Neither caps returned arrays or retained factors: evaluation
    needs O(batch*n*n) output and factorization retains that many LU entries.
    Use smaller external batches if that storage is too large. No full inverse
    or variable-by-variable Hessian is constructed. No positive-definite cone
    membership is implied by a positive determinant.
    """

    def __init__(
        self,
        matrices: Any,
        *,
        backend: str = "numpy",
        point_block_size: int = 64,
        coefficient_block_size: int = 16,
    ) -> None:
        self._point_block = _positive_size(point_block_size, "point_block_size")
        self._coefficient_block = _positive_size(
            coefficient_block_size, "coefficient_block_size"
        )
        self._backend = backend
        self._device: int | None = None
        self._cupyx: Any = None
        if backend == "numpy":
            self._xp: Any = np
            self._linalg: Any = sla
        elif backend == "cupy":
            try:
                self._xp = importlib.import_module("cupy")
                self._cupyx = importlib.import_module("cupyx")
                self._linalg = importlib.import_module("cupyx.scipy.linalg")
            except ImportError as error:
                raise ImportError(
                    "backend='cupy' requires CuPy installed for your CUDA runtime"
                ) from error
            if self._xp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("backend='cupy' requires a working GPU")
            self._device = int(self._xp.cuda.runtime.getDevice())
        else:
            raise ValueError("backend must be 'numpy' or 'cupy'")
        self._validate_array(matrices, "matrices")
        if (
            matrices.ndim != 3
            or matrices.shape[0] == 0
            or matrices.shape[1] == 0
            or matrices.shape[1] != matrices.shape[2]
        ):
            raise ValueError("matrices must have shape (m, n, n) with m,n > 0")
        self._m, self._n, _ = matrices.shape
        self._coefficients = self._xp.array(matrices, copy=True, order="C")
        # Each coefficient occupies n adjacent columns of the reusable RHS.
        self._rhs = self._xp.ascontiguousarray(
            self._coefficients.transpose(1, 0, 2)
        ).reshape(self.n, self.m * self.n)

    @property
    def m(self) -> int:
        return int(self._m)

    @property
    def n(self) -> int:
        return int(self._n)

    @property
    def backend(self) -> str:
        return self._backend

    def _check_device(self) -> None:
        if self._device is not None:
            if self._xp.cuda.runtime.getDevice() != self._device:
                raise ValueError("Current device must match the preparation device")

    def _validate_array(self, array: Any, name: str) -> None:
        self._check_device()
        if not isinstance(array, self._xp.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a resident {self.backend} float64 ndarray")
        if self._device is not None and array.device.id != self._device:
            raise ValueError(f"{name} must reside on the preparation device")
        if not bool(self._xp.isfinite(array).all()):
            raise ValueError(f"{name} must contain only finite values")

    def _points(self, points: Any) -> tuple[Any, tuple[int, ...]]:
        self._validate_array(points, "points")
        if points.ndim < 1 or points.shape[-1] != self.m:
            raise ValueError(f"points must have shape (..., {self.m})")
        return points.reshape(-1, self.m), points.shape[:-1]

    def _finite_result(self, value: Any) -> None:
        if not bool(self._xp.isfinite(value).all()):
            raise FloatingPointError("Nonfinite numerical result; rescale the pencil")

    def _assemble(self, points: Any) -> Any:
        with np.errstate(over="ignore", invalid="ignore"):
            result = (points @ self._coefficients.reshape(self.m, -1)).reshape(
                -1, self.n, self.n
            )
        self._finite_result(result)
        return result

    def evaluate(self, points: Any) -> Any:
        """Return owned matrices with shape ``(..., n, n)`` on this backend."""
        flat, shape = self._points(points)
        result = self._xp.empty((len(flat), self.n, self.n), dtype=np.float64)
        for start in range(0, len(flat), self._point_block):
            stop = start + self._point_block
            result[start:stop] = self._assemble(flat[start:stop])
        return result.reshape(*shape, self.n, self.n)

    def factor(self, points: Any) -> PencilFactorization:
        """Own LU factors for a point snapshot, reusable across derivative queries.

        Singular matrices are retained for slogdet but all derivative queries
        reject a batch containing one. No tolerance truncates near-singular
        matrices; their derivatives remain conditioning-dependent.
        """
        flat, shape = self._points(points)
        xp = self._xp
        factors: list[tuple[Any, Any]] = []
        signs = xp.empty(len(flat), dtype=np.float64)
        logs = xp.empty(len(flat), dtype=np.float64)
        indices = xp.arange(self.n)
        for start in range(0, len(flat), self._point_block):
            matrices = self._assemble(flat[start : start + self._point_block])
            for offset, matrix in enumerate(matrices):
                # Positive LU info denotes singularity, a valid slogdet result.
                context = (
                    self._cupyx.errstate(linalg="ignore")
                    if self._cupyx is not None
                    else nullcontext()
                )
                with context, warnings.catch_warnings():
                    warnings.simplefilter("ignore", sla.LinAlgWarning)
                    lu, piv = self._linalg.lu_factor(matrix, check_finite=False)
                self._finite_result(lu)
                factors.append((lu, piv))
                diagonal = xp.diagonal(lu)
                parity = xp.count_nonzero(piv != indices) % 2
                signs[start + offset] = (1 - 2 * parity) * xp.prod(xp.sign(diagonal))
                with np.errstate(divide="ignore"):
                    logs[start + offset] = xp.sum(xp.log(xp.abs(diagonal)))
        return PencilFactorization(self, factors, signs, logs, shape)


class PencilFactorization:
    """Reusable result of :meth:`PreparedMatrixPencil.factor`.

    Owns LU factors, not input points. Returned arrays are independent copies.
    Slogdet and gradient are cached; HVPs reuse factors and compute coefficient
    solves in bounded blocks. A combined first gradient/HVP query uses one pass
    of those solves. This trades recomputation on subsequent HVPs for bounded
    O(coefficient_block*n*n) scratch rather than O(batch*m*n*n) retained solves.
    """

    def __init__(
        self,
        pencil: PreparedMatrixPencil,
        factors: list[tuple[Any, Any]],
        signs: Any,
        logs: Any,
        shape: tuple[int, ...],
    ) -> None:
        self._pencil = pencil
        self._factors = factors
        self._signs = signs
        self._logs = logs
        self._shape = shape
        self._gradient: Any = None
        self._singular = bool(pencil._xp.any(signs == 0))

    def slogdet(self) -> tuple[Any, Any]:
        """Return sign and log(abs(det)); singular entries yield (0, -inf)."""
        self._pencil._check_device()
        return (
            self._signs.reshape(self._shape).copy(),
            self._logs.reshape(self._shape).copy(),
        )

    def _solve(self, factor: tuple[Any, Any], rhs: Any) -> Any:
        p = self._pencil
        context = (
            p._cupyx.errstate(linalg="raise") if p._cupyx is not None else nullcontext()
        )
        with context:
            result = p._linalg.lu_solve(factor, rhs, check_finite=False)
        p._finite_result(result)
        return result

    def _derivatives(self, directions: Any = None) -> Any:
        p = self._pencil
        p._check_device()
        if self._singular:
            raise np.linalg.LinAlgError(
                "Logdet derivatives require nonsingular matrices"
            )
        xp = p._xp
        need_gradient = self._gradient is None
        gradient = (
            xp.empty((len(self._factors), p.m), dtype=np.float64)
            if need_gradient
            else self._gradient
        )
        hvp = xp.empty_like(gradient) if directions is not None else None
        for row, factor in enumerate(self._factors):
            bv = None
            if directions is not None:
                av = p._assemble(directions[row : row + 1])[0]
                bv = self._solve(factor, av)
            for start in range(0, p.m, p._coefficient_block):
                stop = min(start + p._coefficient_block, p.m)
                solved = self._solve(factor, p._rhs[:, start * p.n : stop * p.n])
                blocks = solved.reshape(p.n, stop - start, p.n).transpose(1, 0, 2)
                if need_gradient:
                    gradient[row, start:stop] = xp.trace(blocks, axis1=1, axis2=2)
                if hvp is not None:
                    hvp[row, start:stop] = -xp.einsum("kij,ji->k", blocks, bv)
        p._finite_result(gradient)
        if hvp is not None:
            p._finite_result(hvp)
        self._gradient = gradient
        return hvp

    def logabsdet_gradient(self) -> Any:
        r"""Return ``tr(A(x)^-1 A_i)`` with shape ``(..., m)``.

        Requires nonsingular matrices, including outside the positive-definite
        cone. Near-singular inputs are not regularized or pseudoinverted.
        """
        self._pencil._check_device()
        if self._gradient is None:
            self._derivatives()
        return self._gradient.reshape(*self._shape, self._pencil.m).copy()

    def logabsdet_hvp(self, directions: Any) -> Any:
        r"""Return ``-tr(A^-1 A_i A^-1 A(v))``, without a full Hessian.

        Directions must be resident finite float64 arrays of shape ``(m,)``
        (one shared direction) or exactly ``points.shape``. No other implicit
        broadcasting is accepted. The first HVP also caches the gradient.
        """
        p = self._pencil
        p._validate_array(directions, "directions")
        if directions.shape == (p.m,):
            flat = p._xp.broadcast_to(directions, (len(self._factors), p.m))
        elif directions.shape == (*self._shape, p.m):
            flat = directions.reshape(-1, p.m)
        else:
            raise ValueError("directions must have shape (m,) or the points shape")
        return self._derivatives(flat).reshape(*self._shape, p.m)
