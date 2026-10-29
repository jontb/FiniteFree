"""Bounded-storage empirical coefficient and interval-count diagnostics."""

import operator
from typing import Any, Callable, Sequence, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from .core import RealRootedPolynomial


def _sample_observations(
    matrix: Any, degree: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the retained comparison's Hermitian/paired-eigenvalue contract."""
    raw = np.asarray(matrix)
    # Real samples keep the real symmetric eigensolver; complex/object input is
    # converted as before.
    sample = np.asarray(
        raw, dtype=np.float64 if raw.dtype.kind in "biuf" else np.complex128
    )
    if sample.shape not in [(degree, degree), (2 * degree, 2 * degree)]:
        raise ValueError("Sample matrices must have shape (d, d) or (2d, 2d).")
    if not np.all(np.isfinite(sample)):
        raise ValueError("Sample matrices must contain finite entries.")
    scale = np.max(np.abs(sample), initial=0.0)
    if not np.isfinite(scale) or not np.allclose(
        sample, sample.conj().T, rtol=0, atol=1e-10 * scale
    ):
        raise ValueError("Sample matrices must be numerically Hermitian.")
    eigenvalues = np.linalg.eigvalsh(sample)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Sample eigenvalues must be finite.")
    if degree and sample.shape[0] == 2 * degree:
        if not np.allclose(
            eigenvalues[::2],
            eigenvalues[1::2],
            rtol=0,
            atol=1e-10 * np.max(np.abs(eigenvalues)),
        ):
            raise ValueError("Doubled-size samples must have paired eigenvalues.")
        eigenvalues = eigenvalues[::2]
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = np.atleast_1d(np.poly(eigenvalues))
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Sample characteristic coefficients must be finite.")
    return np.asarray(eigenvalues, dtype=np.float64), np.asarray(
        coefficients, dtype=np.float64
    )


class StreamingEmpiricalComparison:
    """Accumulate coefficient moments and counts without retaining samples.

    update(matrix) consumes one Hermitian sample; extend(generator, samples)
    consumes a batch. Intervals are closed [a,b]. Storage is O(d + m*d) for
    degree d and m intervals, independent of the sample count. Dynamically
    scaled Welford moments avoid squaring large/tiny raw coefficients.
    """

    def __init__(
        self,
        analytical_poly: RealRootedPolynomial,
        intervals: Sequence[tuple[float, float]] = (),
    ) -> None:
        self._polynomial = RealRootedPolynomial(
            analytical_poly.coeffs,
            monic=False,
            assume_real_rooted=analytical_poly._is_verified,
        )
        self.d = analytical_poly.degree
        limits = np.asarray(intervals)
        if np.iscomplexobj(limits):
            raise ValueError("Intervals must be finite real pairs with a <= b")
        try:
            limits = np.array(limits, dtype=float, copy=True).reshape((-1, 2))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Intervals must be finite real pairs with a <= b"
            ) from error
        if (
            (len(intervals) and np.asarray(intervals).shape != (len(intervals), 2))
            or not np.all(np.isfinite(limits))
            or np.any(limits[:, 0] > limits[:, 1])
        ):
            raise ValueError("Intervals must be finite real pairs with a <= b")
        self._intervals = limits
        self._samples = 0
        self._scale = np.zeros(self.d + 1)
        self._mean = np.zeros(self.d + 1)
        self._m2 = np.zeros(self.d + 1)
        self._counts = np.zeros((len(limits), self.d + 1), dtype=np.int64)

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def intervals(self) -> NDArray[np.float64]:
        return self._intervals.copy()

    @property
    def count_histograms(self) -> NDArray[np.int64]:
        """One integer histogram on 0..d for each configured interval."""
        return self._counts.copy()

    @property
    def coefficient_mean(self) -> NDArray[np.float64]:
        if not self.samples:
            raise ValueError("At least one sample is required")
        return self._mean * self._scale

    @property
    def coefficient_std(self) -> NDArray[np.float64]:
        """Sample standard deviation (ddof=1), if representable in float64."""
        if self.samples < 2:
            raise ValueError("At least two samples are required")
        with np.errstate(over="ignore"):
            result = np.sqrt(self._m2 / (self.samples - 1)) * self._scale
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Coefficient standard deviation exceeds float64 range")
        return result

    @property
    def count_mean(self) -> NDArray[np.float64]:
        if not self.samples:
            raise ValueError("At least one sample is required")
        mean: NDArray[np.float64] = (self._counts / self.samples) @ np.arange(
            self.d + 1, dtype=float
        )
        return mean

    @property
    def count_variance(self) -> NDArray[np.float64]:
        """Sample variance (ddof=1) of interval counts."""
        if self.samples < 2:
            raise ValueError("At least two samples are required")
        offsets = np.arange(self.d + 1) - self.count_mean[:, None]
        variance: NDArray[np.float64] = np.sum(self._counts * offsets**2, axis=1) / (
            self.samples - 1
        )
        return variance

    def update(self, matrix: Any) -> None:
        """Add one sample atomically; invalid samples leave all statistics intact."""
        eigenvalues, coefficients = _sample_observations(matrix, self.d)
        counts = np.sum(
            (eigenvalues >= self._intervals[:, :1])
            & (eigenvalues <= self._intervals[:, 1:]),
            axis=1,
        )
        new_scale = np.maximum(self._scale, np.abs(coefficients))
        safe_scale = np.where(new_scale == 0, 1.0, new_scale)
        ratio = self._scale / safe_scale
        mean = self._mean * ratio
        m2 = self._m2 * ratio**2
        observation = coefficients / safe_scale
        delta = observation - mean
        mean = mean + delta / (self.samples + 1)
        m2 = m2 + delta * (observation - mean)
        self._scale, self._mean, self._m2 = new_scale, mean, np.maximum(m2, 0)
        self._samples += 1
        self._counts[np.arange(len(counts)), counts] += 1

    def extend(
        self, generator: Callable[[], NDArray[Any]], samples: SupportsIndex
    ) -> None:
        """Add nonnegative integer samples; prior successful updates are retained."""
        try:
            count = operator.index(samples)
        except TypeError as error:
            raise ValueError("samples must be a non-negative integer") from error
        if isinstance(samples, (bool, np.bool_)) or count < 0:
            raise ValueError("samples must be a non-negative integer")
        for _ in range(count):
            self.update(generator())

    def verify_coefficients(
        self, alpha: float = 0.05, *, rtol: float = 1e-10, atol: float = 0.0
    ) -> bool:
        """Use the retained comparison's Bonferroni t-band assumptions.

        Independent samples are required. Bands are exact for normal
        coefficient observations and otherwise use a large-sample approximation.
        This diagnostic is not a proof of the generator's distribution.
        """
        from scipy.stats import t

        if self.samples < 2:
            raise ValueError("At least two samples are required")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be finite and in the open interval (0, 1).")
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be finite and non-negative.")
        try:
            target = np.array(self._polynomial.coeffs, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Analytical coefficients must be finite in float64."
            ) from error
        if not np.all(np.isfinite(target)):
            raise ValueError("Analytical coefficients must be finite in float64.")
        critical = t.isf(alpha / (2 * (self.d + 1)), self.samples - 1)
        if not np.isfinite(critical):
            raise ValueError("alpha is too small for a finite float64 critical value.")
        scale = np.maximum(self._scale, np.abs(target))
        scale = np.where(scale == 0, 1.0, scale)
        mean = self._mean * (self._scale / scale)
        sem = np.sqrt(self._m2 / (self.samples - 1) / self.samples) * (
            self._scale / scale
        )
        target = target / scale
        with np.errstate(over="ignore"):
            tolerance = atol / scale + rtol * np.abs(target)
        return bool(np.all(np.abs(mean - target) <= critical * sem + tolerance))
