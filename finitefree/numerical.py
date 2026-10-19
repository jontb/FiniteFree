"""Optional float64 batch kernels and finite-rank DPP count laws.

These APIs do not change exact scalar kernels. Continuous restrictions use
Gauss-Legendre quadrature: their probabilities are estimates, not enclosures.
"""

import math
import operator
from decimal import Decimal, localcontext
from typing import Any, Callable, Optional, Sequence, SupportsIndex

import flint
import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

from .dpp import BaseKernel, OrthogonalPolynomialKernel


def _real_array(value: Any, name: str) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or (
        raw.dtype == object and any(np.iscomplexobj(x) for x in raw.flat)
    ):
        raise ValueError(f"{name} must contain finite real values")
    try:
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite real values") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real values")
    return result


def _decimal_real(value: Any) -> Decimal:
    try:
        result = (
            Decimal(int(value.p)) / Decimal(int(value.q))
            if isinstance(value, flint.fmpq)
            else Decimal(str(value.evalf(40)))
        )
        if not result.is_finite():
            raise ValueError("nonfinite basis value")
        return result
    except (ArithmeticError, ValueError, AttributeError) as error:
        raise ValueError("Basis data must be finite real numerical values") from error


class PreparedOrthogonalKernel:
    """Owned numerical basis, reusable across batches and interval restrictions.

    Verified Hermite bases use the normalized recurrence. Other bases use
    normalized polynomial coefficients and Horner evaluation. Neither path is
    an arbitrary-precision calculation; nonfinite results raise. Underflow and
    ill-conditioning can still lose information in finite float64 results.
    """

    def __init__(self, kernel: OrthogonalPolynomialKernel) -> None:
        self.n = kernel.n
        self._hermite = kernel._hermite_parameters is not None
        self._center = kernel._hermite_center
        self._initial = kernel._hermite_initial
        self._steps = kernel._hermite_steps
        self._coefficients: tuple[NDArray[np.float64], ...] = ()
        if self._hermite:
            if kernel._hermite_range_error:
                raise ValueError("Hermite normalization exceeds finite float64 range")
            return
        normalized = []
        for polynomial, norm in zip(kernel.polys[: self.n], kernel.norms):
            # Rational coefficients are normalized before narrowing, so a large
            # polynomial scaling can cancel a large norm without overflow.
            with localcontext() as context:
                context.prec = 40
                mass = _decimal_real(norm)
                if mass <= 0:
                    raise ValueError("Basis norms must be finite and positive")
                root = mass.sqrt()
                values = []
                for coefficient in polynomial.coeffs:
                    values.append(float(_decimal_real(coefficient) / root))
                normalized.append(_real_array(values, "Normalized coefficients"))
        self._coefficients = tuple(normalized)

    def basis(self, points: Any) -> NDArray[np.float64]:
        """Return normalized basis rows, shape (number of points, rank)."""
        xs = _real_array(points, "Coordinates")
        if xs.ndim != 1:
            raise ValueError("Coordinates must be one-dimensional")
        result = np.empty((len(xs), self.n))
        with np.errstate(over="ignore", invalid="ignore"):
            if self._hermite and self.n:
                previous = np.zeros(len(xs))
                current = np.full(len(xs), self._initial)
                result[:, 0] = current
                shifted = xs - self._center
                for j, (inverse_sqrt, ratio) in enumerate(self._steps, 1):
                    previous, current = (
                        current,
                        shifted * inverse_sqrt * current - ratio * previous,
                    )
                    result[:, j] = current
            else:
                for j, coefficients in enumerate(self._coefficients):
                    result[:, j] = np.polyval(coefficients, xs)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Numerical basis exceeds finite float64 range")
        return result

    def matrix(self, xs: Any, ys: Any = None) -> NDArray[np.float64]:
        """Return a float64 Gram/cross-kernel matrix, including empty batches."""
        left = self.basis(xs)
        right = left if ys is None else self.basis(ys)
        with np.errstate(over="ignore", invalid="ignore"):
            result = left @ right.T
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Numerical kernel exceeds finite float64 range")
        return result

    def quadrature(
        self,
        a: float,
        b: float,
        n_points: SupportsIndex = 50,
        weight_func: Optional[Callable[[float], float]] = None,
    ) -> "PreparedKernelQuadrature":
        """Prepare a finite interval restriction in the kernel's measure."""
        return PreparedKernelQuadrature(self, a, b, n_points, weight_func)


class PreparedKernelQuadrature:
    """Owned quadrature nodes, measure weights and rank-by-rank restricted Gram.

    weight_func is the density of the kernel's reference measure; it defaults
    to Lebesgue density 1. It is called once per node. No density is inferred
    from a polynomial family. Storage is O(n_points * rank + rank**2).
    """

    def __init__(
        self,
        kernel: PreparedOrthogonalKernel,
        a: float,
        b: float,
        n_points: SupportsIndex = 50,
        weight_func: Optional[Callable[[float], float]] = None,
    ) -> None:
        if not math.isfinite(a) or not math.isfinite(b) or a >= b:
            raise ValueError("Interval endpoints must be finite with a < b")
        try:
            order = operator.index(n_points)
        except TypeError as error:
            raise ValueError("n_points must be a positive integer") from error
        if isinstance(n_points, (bool, np.bool_)) or order < 1:
            raise ValueError("n_points must be a positive integer")
        points, weights = roots_legendre(order)
        # Half-endpoint arithmetic avoids overflowing b-a or a+b.
        half_width = b / 2 - a / 2
        self._nodes: NDArray[np.float64] = half_width * points + (a / 2 + b / 2)
        density = (
            np.ones(order)
            if weight_func is None
            else _real_array([weight_func(float(x)) for x in self._nodes], "Density")
        )
        if density.shape != (order,) or np.any(density < 0):
            raise ValueError("Density must return finite non-negative scalar values")
        with np.errstate(over="ignore", invalid="ignore"):
            self._weights: NDArray[np.float64] = half_width * weights * density
            self._weighted_basis = (
                kernel.basis(self._nodes) * np.sqrt(self._weights)[:, None]
            )
            self._gram: NDArray[np.float64] = (
                self._weighted_basis.T @ self._weighted_basis
            )
        if not np.all(np.isfinite(self._weights)) or not np.all(
            np.isfinite(self._gram)
        ):
            raise RuntimeError("Quadrature exceeds finite float64 range")

    @property
    def nodes(self) -> NDArray[np.float64]:
        return self._nodes.copy()

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights.copy()

    @property
    def gram(self) -> NDArray[np.float64]:
        return self._gram.copy()

    def count_distribution(self) -> "DPPCountDistribution":
        """Compute a count law from this restriction without reevaluating it."""
        return DPPCountDistribution(self._gram, estimated=True)

    def gap_probability(self) -> float:
        """Estimated probability of no points, using the same restricted Gram."""
        return float(self.count_distribution().pmf(0))


class DPPCountDistribution:
    """Poisson-binomial count law from a real symmetric restricted DPP kernel.

    Eigenvalues outside [0,1] by more than 1e-12 are rejected. Only roundoff
    within this absolute tolerance is clipped. spectral_correction and
    probability_mass_error expose those numerical adjustments. estimated marks
    quadrature input; even unmarked float64 results are not certified bounds.
    """

    def __init__(self, restricted_matrix: Any, *, estimated: bool = False) -> None:
        matrix = _real_array(restricted_matrix, "Restricted kernel")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Restricted kernel must be square")
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
            raise ValueError("Restricted kernel must be symmetric")
        eigenvalues = np.linalg.eigvalsh(matrix / 2 + matrix.T / 2)
        if np.any(eigenvalues < -1e-12) or np.any(eigenvalues > 1 + 1e-12):
            raise ValueError("Restricted kernel eigenvalues must lie in [0, 1]")
        self._eigenvalues: NDArray[np.float64] = np.clip(eigenvalues, 0, 1)
        self.spectral_correction = float(
            np.max(np.abs(self._eigenvalues - eigenvalues), initial=0)
        )
        self.estimated = bool(estimated)
        pmf = np.ones(1)
        for probability in self._eigenvalues:
            pmf = np.convolve(pmf, [1 - probability, probability])
        mass = math.fsum(pmf)
        self.probability_mass_error = abs(mass - 1)
        self._pmf = pmf / mass

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        return self._eigenvalues.copy()

    @property
    def mean(self) -> float:
        return math.fsum(self._eigenvalues)

    @property
    def variance(self) -> float:
        return math.fsum(self._eigenvalues * (1 - self._eigenvalues))

    def pmf(self, k: Optional[SupportsIndex] = None) -> Any:
        """P(N=k), or a caller-owned array on 0..rank when k is omitted."""
        if k is None:
            return self._pmf.copy()
        index = operator.index(k)
        return float(self._pmf[index]) if 0 <= index < len(self._pmf) else 0.0

    def cdf(self, k: SupportsIndex) -> float:
        """P(N<=k) for an integer threshold."""
        index = operator.index(k)
        if index < 0:
            return 0.0
        if index >= len(self._pmf) - 1:
            return 1.0
        return min(1.0, math.fsum(self._pmf[: index + 1]))

    def sf(self, k: SupportsIndex) -> float:
        """P(N>k), summed directly to avoid cancellation in 1-cdf(k)."""
        index = operator.index(k)
        if index < 0:
            return 1.0
        return min(1.0, math.fsum(self._pmf[index + 1 :]))


def count_distribution_discrete(
    kernel: BaseKernel, points: Sequence[Any]
) -> DPPCountDistribution:
    """Numerical count law on distinct states of a discrete correlation kernel."""
    if len(set(points)) != len(points):
        raise ValueError("points must contain distinct states")
    matrix = kernel.matrix(points) if len(points) else np.empty((0, 0))
    return DPPCountDistribution(matrix)


def count_distribution_continuous(
    kernel: OrthogonalPolynomialKernel,
    a: float,
    b: float,
    n_points: SupportsIndex = 50,
    weight_func: Optional[Callable[[float], float]] = None,
) -> DPPCountDistribution:
    """Estimated finite-rank interval count law; pass the reference density."""
    return (
        kernel.prepare_numeric()
        .quadrature(a, b, n_points, weight_func)
        .count_distribution()
    )
