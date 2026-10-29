"""Optional float64 batch kernels and finite-rank DPP count laws.

These APIs do not change exact scalar kernels. Continuous restrictions use
Gauss-Legendre quadrature: their probabilities are estimates, not enclosures.
"""

import math
import operator
from copy import copy
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


def _decimal_sqrt_float(value: Any) -> float:
    """Positive float sqrt of a positive fmpq, rounded after a 40-digit root."""
    with localcontext() as context:
        context.prec = 40
        result = float((Decimal(int(value.p)) / Decimal(int(value.q))).sqrt())
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Recurrence normalization exceeds float64 range")
    return result


def _verified_recurrence(kernel: OrthogonalPolynomialKernel) -> Any:
    """Exact monic three-term recurrence data for the supplied basis, or None.

    Checks q_{j+1} = (x - a_j) q_j - b_j q_{j-1} exactly for the monic
    q_j = p_j / k_j through q_n, with rational a_j and b_j, and the norm ratios
    h_j / k_j**2 = b_j h_{j-1} / k_{j-1}**2 that orthonormality implies. No
    family provenance or caller-supplied leading coefficients are trusted.
    Returns (signs, monic h_0, a_0..a_{n-1}, b_0..b_{n-1}) with b_0 = 0.
    """
    n = kernel.n
    if n == 0:
        return None
    monic = []
    signs = []
    monic_norms = []
    for j, polynomial in enumerate(kernel._polys[: n + 1]):
        if not polynomial._is_flint or polynomial.degree != j:
            return None
        leading = polynomial._fmpq_poly[j]
        monic.append(polynomial._fmpq_poly / leading)
        if j < n:
            norm = kernel._norms[j]
            if not isinstance(norm, flint.fmpq) or norm <= 0:
                return None
            signs.append(1.0 if leading > 0 else -1.0)
            monic_norms.append(norm / leading**2)
    x = flint.fmpq_poly([0, 1])
    shifts = []
    products = [flint.fmpq(0)]
    for j in range(n):
        # Remainder a_j q_j + b_j q_{j-1}; q_j is monic, so a_j is its x**j term.
        remainder = x * monic[j] - monic[j + 1]
        shift = remainder[j]
        remainder -= shift * monic[j]
        if j == 0:
            if not remainder.is_zero():
                return None
        else:
            product = remainder[j - 1]
            if (
                product <= 0
                or remainder != product * monic[j - 1]
                or monic_norms[j] != product * monic_norms[j - 1]
            ):
                return None
            products.append(product)
        shifts.append(shift)
    return signs, monic_norms[0], shifts, products


class PreparedOrthogonalKernel:
    """Owned numerical basis, reusable across batches and interval restrictions.

    A basis with an exactly verified monic three-term recurrence and consistent
    norms (for example shifted/scaled/signed Hermite, Legendre, Laguerre or
    Jacobi data) uses the orthonormal float recurrence, then applies the sign of
    each supplied leading coefficient. Other bases use normalized polynomial
    coefficients and Horner evaluation. Neither path is an arbitrary-precision
    calculation; nonfinite results raise. Underflow and ill-conditioning can
    still lose information in finite float64 results.
    """

    def __init__(self, kernel: OrthogonalPolynomialKernel) -> None:
        self.n = kernel.n
        if kernel._hermite_parameters is not None and kernel._hermite_range_error:
            raise ValueError("Hermite normalization exceeds finite float64 range")
        # psi_{j+1} = (x - shift_j) * scale_j * psi_j - ratio_j * psi_{j-1}.
        self._recurrence = False
        self._initial = 1.0
        self._steps: tuple[tuple[float, float, float], ...] = ()
        self._signs: NDArray[np.float64] = np.ones(self.n)
        self._coefficients: tuple[NDArray[np.float64], ...] = ()
        verified = _verified_recurrence(kernel)
        if verified is not None:
            signs, mass, shifts, products = verified
            try:
                # Square roots of exact ratios precede float conversion, so
                # large basis scalings and norms cancel without overflow.
                self._initial = _decimal_sqrt_float(1 / mass)
                self._steps = tuple(
                    (
                        float(_decimal_real(shifts[j])),
                        _decimal_sqrt_float(1 / products[j + 1]),
                        _decimal_sqrt_float(products[j] / products[j + 1])
                        if j
                        else 0.0,
                    )
                    for j in range(self.n - 1)
                )
                if all(math.isfinite(step[0]) for step in self._steps):
                    self._signs = np.array(signs)
                    self._recurrence = True
                    return
            except (OverflowError, ValueError):
                pass
            # Unrepresentable recurrence data keeps the coefficient path.
            self._initial, self._steps = 1.0, ()
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
            if self._recurrence:
                previous = np.zeros(len(xs))
                current = np.full(len(xs), self._initial)
                result[:, 0] = current
                center = None
                for j, (shift, scale, ratio) in enumerate(self._steps, 1):
                    if shift != center:  # Constant for Hermite/symmetric families.
                        center, shifted = shift, xs - shift
                    previous, current = (
                        current,
                        shifted * scale * current - ratio * previous,
                    )
                    result[:, j] = current
                # Positive-leading orthonormal columns take the supplied signs.
                if np.any(self._signs < 0):
                    result *= self._signs
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
    from a polynomial family. The first successful count/gap query caches its
    count law; returned laws own their arrays and metadata. Validation remains
    lazy. Storage is O(n_points * rank + rank**2), including the O(rank) cache.
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
        self._cached_distribution: Optional[DPPCountDistribution] = None

    @property
    def nodes(self) -> NDArray[np.float64]:
        return self._nodes.copy()

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights.copy()

    @property
    def gram(self) -> NDArray[np.float64]:
        return self._gram.copy()

    def _count_law(self) -> "DPPCountDistribution":
        if self._cached_distribution is None:
            self._cached_distribution = DPPCountDistribution(self._gram, estimated=True)
        return self._cached_distribution

    def count_distribution(self) -> "DPPCountDistribution":
        """Return an owned count law, reusing its successful first computation."""
        return self._count_law()._copy()

    def gap_probability(self) -> float:
        """Estimated probability of no points, using the same restricted Gram."""
        return float(self._count_law().pmf(0))


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

    def _copy(self) -> "DPPCountDistribution":
        result = copy(self)
        result._eigenvalues = self._eigenvalues.copy()
        result._pmf = self._pmf.copy()
        return result

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
