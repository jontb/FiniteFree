import math
import numbers
import operator
from typing import Any, Callable, Optional, SupportsFloat, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from .core import RealRootedPolynomial
from .streaming import (
    _has_adjacent_pairs,
    _is_numerically_hermitian,
    _sample_observations,
)
from .transforms import _moment_index
from .utils.roots import _gaussian_roots, _laguerre_roots


def _normal_draw(
    rng: Optional[np.random.Generator],
) -> Callable[..., NDArray[np.float64]]:
    """Select a stream without changing the legacy draw order or global state."""
    if rng is None:
        return np.random.randn
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")
    return lambda *shape: rng.standard_normal(shape)


def sample_gue(
    d: int, scale: float = 1.0, *, rng: Optional[np.random.Generator] = None
) -> Any:
    r"""
    Generates a sample GUE matrix (Hermitian, Gaussian entries)
    with diagonal and off-diagonal variance of $1/d$.
    """
    normal = _normal_draw(rng)
    X = (normal(d, d) + 1j * normal(d, d)) / np.sqrt(2.0)
    H = (X + X.conj().T) / (np.sqrt(2.0) * np.sqrt(d))
    return H * scale


def sample_goe(
    d: int, scale: float = 1.0, *, rng: Optional[np.random.Generator] = None
) -> Any:
    r"""
    Generates a sample GOE matrix (Symmetric, Gaussian entries)
    with diagonal variance $2/d$ and off-diagonal variance $1/d$.
    """
    normal = _normal_draw(rng)
    X = normal(d, d)
    H = (X + X.T) / (np.sqrt(2.0) * np.sqrt(d))
    return H * scale


def sample_gse(
    d: int, scale: float = 1.0, *, rng: Optional[np.random.Generator] = None
) -> Any:
    r"""
    Generates a sample GSE matrix of dimension $2d \times 2d$ (Self-dual, Gaussian entries)
    with Kramers degeneracy, scaled to match the $d$-dimensional expected characteristic polynomial.
    """
    normal = _normal_draw(rng)
    X = (normal(2 * d, 2 * d) + 1j * normal(2 * d, 2 * d)) / np.sqrt(2.0)
    H = (X + X.conj().T) / 2.0
    J = np.zeros((2 * d, 2 * d))
    J[:d, d:] = np.eye(d)
    J[d:, :d] = -np.eye(d)
    H = (H + J @ H.conj() @ J.T) / 2.0
    return H * np.sqrt(2.0 / d) * scale


def sample_wishart(
    d: int,
    n: int,
    beta: int = 2,
    scale: float = 1.0,
    *,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[Any]:
    r"""
    Sample a real/complex/quaternionic Wishart representation.
    For beta=1 or 2, return a d-by-d matrix XX^H/n. For beta=4, the complex
    representation has shape (2d,2d), paired eigenvalues, and normalization
    XX^H/(2n). scale multiplies the matrix. beta must be 1, 2 or 4.
    """
    normal = _normal_draw(rng)
    if beta == 1:
        X = normal(d, n)
        W = (X @ X.T) / n
    elif beta == 2:
        X = (normal(d, n) + 1j * normal(d, n)) / np.sqrt(2.0)
        W = (X @ X.conj().T) / n
    elif beta == 4:
        A = (normal(d, n) + 1j * normal(d, n)) / np.sqrt(2.0)
        B = (normal(d, n) + 1j * normal(d, n)) / np.sqrt(2.0)
        X = np.block([[A, B], [-np.conj(B), np.conj(A)]])
        W = (X @ X.conj().T) / (2 * n)
    else:
        raise ValueError("beta must be 1, 2, or 4")
    return W * scale


def matrix_finite_cumulants(
    matrix: Any,
    *,
    d: Optional[SupportsIndex] = None,
    paired: bool = False,
) -> NDArray[np.float64]:
    r"""
    Return the float64 finite cumulants $\kappa_1,\ldots,\kappa_d$ of one sample.
    Validation, pairing and the ambient dimension follow matrix_moments: d-n zero
    eigenvalues are embedded, and the result matches FiniteRTransform(p, order=d,
    d=d) for the effective characteristic polynomial p. The spectrum is centered
    at $\kappa_1$ (the mean over d) before forming normalized coefficients from
    eigenvalues, so higher cumulants are unaffected by large offsets; $\kappa_1$
    is restored. Results are float64 estimates without enclosures; high orders
    are ill-conditioned. Nonfinite cumulants raise RuntimeError. Per-sample
    cumulants, their average and cumulants of an averaged polynomial are
    different targets. Returns a new array.
    """
    eigenvalues, dimension = _sample_spectrum(matrix, d, paired)
    roots = np.zeros(dimension, dtype=np.float64)
    roots[: eigenvalues.shape[0]] = eigenvalues
    mean = math.fsum(roots) / dimension
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = np.poly(roots - mean)
        # Normalized elementary coefficients of the centered roots.
        normalized = [
            (-1) ** k * coefficients[k] / math.comb(dimension, k)
            for k in range(dimension + 1)
        ]
        cumulants = np.empty(dimension, dtype=np.float64)
        classical: list[float] = []
        factor = 1.0  # (-d)^(n-1)/(n-1)!
        for n in range(1, dimension + 1):
            value = normalized[n]
            for k in range(1, n):
                value -= math.comb(n - 1, k - 1) * classical[k - 1] * normalized[n - k]
            classical.append(value)
            if n > 1:
                factor *= -dimension / (n - 1)
            cumulants[n - 1] = value * factor
    cumulants[0] = mean
    if not np.all(np.isfinite(cumulants)):
        raise RuntimeError("Matrix finite cumulants exceed the finite float64 range.")
    return cumulants


def _positive_real(value: Any, name: str, *, nonzero_only: bool = False) -> float:
    """Return a finite real parameter, rejecting booleans and complex values."""
    bound = "nonzero" if nonzero_only else "positive"
    message = f"{name} must be a finite {bound} real number."
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(message)
    result = float(value)
    if not math.isfinite(result) or (result == 0 if nonzero_only else result <= 0):
        raise ValueError(message)
    return result


def sample_beta_hermite(
    d: SupportsIndex,
    beta: SupportsFloat,
    scale: SupportsFloat = 1.0,
    *,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Sample the Dumitriu-Edelman beta-Hermite tridiagonal model in compact form.
    Returns new float64 arrays (diagonal, off_diagonal) of lengths d and d-1 for
    $H = \mathrm{scale}\cdot T/\sqrt{\beta d}$, where T has independent
    $N(0,2)$ diagonal entries and off-diagonal entries
    $\chi_{\beta(d-1)},\ldots,\chi_\beta$ from the top. The eigenvalue density of T
    is proportional to $\prod_{i<j}|\lambda_i-\lambda_j|^\beta e^{-\sum\lambda_i^2/4}$.
    This scale gives $E[\operatorname{tr}H^2/d]=1+(2/\beta-1)/d$, matching the GOE,
    GUE and GSE samplers at beta=1, 2, 4, and an expected characteristic
    polynomial equal to gue_expected_poly(d) for every beta. beta may be any
    finite positive real; scale is a finite nonzero real.
    Arguments are validated before any draw. rng=None uses the legacy global
    NumPy RNG; a Generator is advanced in place. Draw order: d standard normals,
    then d-1 chi-square variates with shapes beta*(d-1),...,beta. No dense matrix
    or eigenvalues are formed; use scipy.linalg.eigvalsh_tridiagonal explicitly.
    """
    dimension = _moment_index(d, "d", positive=True)
    shape = _positive_real(beta, "beta")
    factor = _positive_real(scale, "scale", nonzero_only=True) / math.sqrt(
        shape * dimension
    )
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")
    source: Any = np.random if rng is None else rng
    normals = np.asarray(source.standard_normal(dimension), dtype=np.float64)
    degrees = shape * np.arange(dimension - 1, 0, -1, dtype=np.float64)
    chi_squares = np.asarray(
        source.chisquare(degrees) if dimension > 1 else np.empty(0), dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        diagonal = np.sqrt(2.0) * factor * normals
        off_diagonal = factor * np.sqrt(chi_squares)
    if not (np.all(np.isfinite(diagonal)) and np.all(np.isfinite(off_diagonal))):
        raise RuntimeError("beta-Hermite entries exceed the finite float64 range.")
    return diagonal, off_diagonal


def sample_beta_laguerre(
    d: SupportsIndex,
    n: SupportsFloat,
    beta: SupportsFloat,
    scale: SupportsFloat = 1.0,
    *,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Sample the Dumitriu-Edelman beta-Laguerre bidiagonal factor in compact form.
    Returns new float64 arrays (diagonal, subdiagonal) of lengths d and d-1 of a
    lower bidiagonal B with $W = BB^T$. Before scaling by
    $\sqrt{\mathrm{scale}/(\beta n)}$, the diagonal holds
    $\chi_{\beta n},\chi_{\beta(n-1)},\ldots,\chi_{\beta(n-d+1)}$ and the
    subdiagonal $\chi_{\beta(d-1)},\ldots,\chi_\beta$. W has eigenvalue density
    proportional to $\prod_{i<j}|\lambda_i-\lambda_j|^\beta
    \prod\lambda_i^{\beta(n-d+1)/2-1}e^{-\beta n\sum\lambda_i/(2\,\mathrm{scale})}$,
    $E[W]=\mathrm{scale}\,I$ and, for every beta, the expected characteristic
    polynomial $n^{-d}d!(-1)^dL_d^{(n-d)}(nx)$ (wishart_expected_poly at integer
    n, scale one). At integer n >= d and beta=1, 2, 4 the spectrum has the law of
    sample_wishart(d, n, beta), one eigenvalue per pair at beta=4.
    n is a finite real shape with n > d-1; beta and scale are finite positive
    reals. Arguments are validated before any draw. rng=None uses the legacy
    global NumPy RNG; a Generator is advanced in place. Draw order: the d
    diagonal chi-squares, then the d-1 subdiagonal chi-squares, in the orders
    above. Eigenvalues of W are the squared singular values of B; no Gram matrix
    or eigenvalues are formed here.
    """
    dimension = _moment_index(d, "d", positive=True)
    shape = _positive_real(n, "n")
    if not shape > dimension - 1:
        raise ValueError(f"n must exceed d - 1 = {dimension - 1}.")
    index = _positive_real(beta, "beta")
    factor = math.sqrt(_positive_real(scale, "scale") / (index * shape))
    if not math.isfinite(factor) or factor == 0:
        raise ValueError("scale/(beta n) must be a finite positive float64 value.")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")
    diagonal_degrees = index * (shape - np.arange(dimension, dtype=np.float64))
    subdiagonal_degrees = index * np.arange(dimension - 1, 0, -1, dtype=np.float64)
    if not np.all(np.isfinite(diagonal_degrees)) or not np.all(diagonal_degrees > 0):
        raise ValueError(
            "beta-Laguerre chi degrees of freedom must be finite and positive."
        )
    source: Any = np.random if rng is None else rng
    diagonal_squares = np.asarray(source.chisquare(diagonal_degrees), dtype=np.float64)
    subdiagonal_squares = np.asarray(
        source.chisquare(subdiagonal_degrees) if dimension > 1 else np.empty(0),
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        diagonal = factor * np.sqrt(diagonal_squares)
        subdiagonal = factor * np.sqrt(subdiagonal_squares)
    if not (np.all(np.isfinite(diagonal)) and np.all(np.isfinite(subdiagonal))):
        raise RuntimeError("beta-Laguerre entries exceed the finite float64 range.")
    return diagonal, subdiagonal


def sample_haar_unitary(d: int, *, rng: Optional[np.random.Generator] = None) -> Any:
    """Generates a Haar-distributed random unitary matrix."""
    normal = _normal_draw(rng)
    X = (normal(d, d) + 1j * normal(d, d)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(X)
    d_r = np.diagonal(R)
    ph = d_r / np.abs(d_r)
    return Q * ph


def sample_haar_orthogonal(d: int, *, rng: Optional[np.random.Generator] = None) -> Any:
    """Generates a Haar-distributed random orthogonal matrix."""
    normal = _normal_draw(rng)
    X = normal(d, d)
    Q, R = np.linalg.qr(X)
    d_r = np.diagonal(R)
    ph = d_r / np.abs(d_r)
    return Q * ph


def sample_haar_symplectic(
    d: int, *, rng: Optional[np.random.Generator] = None
) -> NDArray[np.complex128]:
    r"""Generates a Haar-distributed random symplectic matrix in $\text{USp}(2d)$."""
    normal = _normal_draw(rng)
    X = (normal(d, d) + 1j * normal(d, d)) / np.sqrt(2)
    Y = (normal(d, d) + 1j * normal(d, d)) / np.sqrt(2)
    Z = np.block([[X, Y], [-np.conj(Y), np.conj(X)]])
    Q = np.zeros((2 * d, 2 * d), dtype=complex)
    J = np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])
    for j in range(d):
        v = Z[:, j]
        if j > 0:
            # Vectorized Gram-Schmidt projection step using NumPy matrix-vector products
            U = Q[:, :j]
            W = Q[:, d : d + j]
            v = v - U @ (U.conj().T @ v) - W @ (W.conj().T @ v)
        u = v / np.linalg.norm(v)
        w = -J @ np.conj(u)
        Q[:, j] = u
        Q[:, j + d] = w
    return Q


def gue_expected_poly(d: int) -> RealRootedPolynomial:
    r"""
    Computes the exact expected characteristic polynomial of a $d \times d$ GUE matrix:
    $\mathbb{E}[\det(xI - M)] = d^{-d/2} He_d(\sqrt{d} x)$
    using the probabilist's Hermite polynomial from orthogonal.py.
    """
    import flint

    from .orthogonal import hermite_polynomial

    he = hermite_polynomial(d, physicist=False)
    f_poly = he._fmpq_poly
    coeffs_list = list(f_poly)
    scaled_coeffs = []
    for i, coeff in enumerate(coeffs_list):
        if coeff == 0:
            scaled_coeffs.append(flint.fmpq(0))
        else:
            power = (i - d) // 2
            factor = flint.fmpq(d) ** power
            scaled_coeffs.append(coeff * factor)

    new_poly = flint.fmpq_poly(scaled_coeffs)
    result = RealRootedPolynomial(new_poly, assume_real_rooted=True)
    return _gaussian_roots(result, flint.fmpq(1, d)) if d else result


def wishart_expected_poly(d: int, n: int, beta: int = 2) -> RealRootedPolynomial:
    r"""
    Construct the monic rational Wishart expectation polynomial.
    For positive integer dimensions with n>=d, coefficients are those of
    $n^{-d}d!(-1)^d L_d^{(n-d)}(nx)$. They match the d eigenvalues used by the
    normalized sampler, including one eigenvalue per pair for beta=4.
    The beta argument is retained but does not change this coefficient formula.
    Known Laguerre recurrence metadata is used only in its supported domain.
    Requires integers d >= 0 and n >= 1; real roots hold for every such pair,
    including n < d, where d - n roots are zero.
    """
    import flint

    from .orthogonal import laguerre_polynomial

    d, n = operator.index(d), operator.index(n)
    if d < 0 or n < 1:
        raise ValueError("wishart_expected_poly requires integers d >= 0 and n >= 1")
    lag = laguerre_polynomial(d, n - d)
    f_poly = lag._fmpq_poly
    coeffs_list = list(f_poly)
    scaled_coeffs = []
    factor_base = flint.fmpq(math.factorial(d) * (-1) ** d)

    for i, coeff in enumerate(coeffs_list):
        if coeff == 0:
            scaled_coeffs.append(flint.fmpq(0))
        else:
            power = i - d
            factor = factor_base * (flint.fmpq(n) ** power)
            scaled_coeffs.append(coeff * factor)

    new_poly = flint.fmpq_poly(scaled_coeffs)
    result = _laguerre_roots(
        RealRootedPolynomial(new_poly, assume_real_rooted=True), n - d
    )
    if result._root_recurrence is not None:
        result._root_dilation = flint.fmpq(1, n)
    return result


def _sample_spectrum(
    matrix: Any, d: Optional[SupportsIndex], paired: bool
) -> tuple[NDArray[np.float64], int]:
    """Validate one Hermitian sample; return effective eigenvalues and ambient d."""
    if not isinstance(paired, (bool, np.bool_)):
        raise ValueError("paired must be a boolean.")
    sample = np.array(matrix, dtype=np.complex128)
    if sample.ndim != 2 or sample.shape[0] != sample.shape[1] or sample.size == 0:
        raise ValueError("matrix must be a nonempty square matrix.")
    if not np.all(np.isfinite(sample)):
        raise ValueError("matrix must contain finite entries.")
    if not _is_numerically_hermitian(sample):
        raise ValueError("matrix must be numerically Hermitian.")
    eigenvalues = np.asarray(np.linalg.eigvalsh(sample), dtype=np.float64)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("matrix eigenvalues must be finite.")
    if paired:
        if sample.shape[0] % 2 or not _has_adjacent_pairs(eigenvalues):
            raise ValueError("paired matrices must have adjacent eigenvalue pairs.")
        eigenvalues = eigenvalues[::2]

    n = eigenvalues.shape[0]
    if d is None:
        return eigenvalues, n
    dimension = _moment_index(d, "d", positive=True)
    if dimension < n:
        raise ValueError(
            f"Ambient dimension d ({dimension}) cannot be less than the "
            f"effective matrix dimension ({n})."
        )
    return eigenvalues, dimension


def matrix_moments(
    matrix: Any,
    order: SupportsIndex,
    *,
    d: Optional[SupportsIndex] = None,
    paired: bool = False,
) -> NDArray[np.float64]:
    r"""
    Return normalized spectral moments $m_0,\ldots,m_{\mathrm{order}}$ of one sample.
    The matrix must be finite, nonempty, square and numerically Hermitian (absolute
    tolerance 1e-10 times the largest entry magnitude); it is not modified. With
    effective dimension n, ambient d >= n (default n) embeds d-n zero eigenvalues:
    $m_0=1$ and $m_k=\operatorname{tr}(M^k)/d$. paired=True requires an even size
    2n with sorted eigenvalues in adjacent pairs (tolerance 1e-10 times the
    largest eigenvalue magnitude), counts one eigenvalue per pair and gives
    $m_k=\operatorname{tr}(M^k)/(2d)$. Even size alone never implies pairing.
    Uses float64 eigenvalue power sums, not characteristic coefficients, so
    coefficient overflow does not affect representable low orders. Results are
    floating estimates with no enclosure; nonfinite moments raise RuntimeError.
    Average the vectors of independent samples to estimate moments of the mean
    spectral measure, which generally differ from the root moments of the mean
    characteristic polynomial. Returns a new float64 array.
    """
    count = _moment_index(order, "order")
    eigenvalues, dimension = _sample_spectrum(matrix, d, paired)
    moments = np.empty(count + 1, dtype=np.float64)
    moments[0] = 1.0
    powers = np.ones_like(eigenvalues)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, count + 1):
            powers = powers * eigenvalues
            moments[k] = np.sum(powers) / dimension
    if not np.all(np.isfinite(moments)):
        raise RuntimeError("Matrix moments exceed the finite float64 range.")
    return moments


class EmpiricalComparison:
    """
    Compare a polynomial with characteristic coefficients from matrix samples.
    Samples must be finite Hermitian matrices of size d, or size 2d with paired
    eigenvalues. Coefficient agreement is a statistical diagnostic, not a proof
    of the generator's distribution.
    """

    def __init__(
        self,
        analytical_poly: RealRootedPolynomial,
        samples: SupportsIndex,
        generator: Callable[[], NDArray[Any]],
    ) -> None:
        try:
            sample_count = operator.index(samples)
        except TypeError as error:
            raise ValueError("samples must be an integer of at least 2.") from error
        if sample_count < 2:
            raise ValueError("samples must be an integer of at least 2.")
        self.analytical_poly = analytical_poly
        self.samples = sample_count
        self.generator = generator
        self.d = analytical_poly.degree

        eigs_list = []
        coeffs_list = []
        for _ in range(sample_count):
            eigs, coefficients = _sample_observations(generator(), self.d)

            eigs_list.append(eigs)
            coeffs_list.append(coefficients)

        self.eigenvalues = np.array(eigs_list)
        self.char_poly_coeffs = np.array(coeffs_list)

    def verify_coefficients(
        self, alpha: float = 0.05, *, rtol: float = 1e-10, atol: float = 0.0
    ) -> bool:
        """
        Check every coefficient against a Bonferroni-adjusted two-sided t band.
        The critical value is t.isf(alpha / (2*(d+1)), samples-1). Zero sample
        variance still requires agreement within atol + rtol*abs(target).
        Per-coefficient scaling avoids overflow/underflow in the sample variance.
        The bands assume independent samples and are exact for normal coefficient
        observations; other distributions require a large-sample approximation.
        """
        from scipy.stats import t

        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be finite and in the open interval (0, 1).")
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be finite and non-negative.")
        try:
            analytical_coeffs = np.array(self.analytical_poly.coeffs, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Analytical coefficients must be finite in float64."
            ) from error
        if not np.all(np.isfinite(analytical_coeffs)):
            raise ValueError("Analytical coefficients must be finite in float64.")
        critical = t.isf(alpha / (2 * (self.d + 1)), self.samples - 1)
        if not np.isfinite(critical):
            raise ValueError("alpha is too small for a finite float64 critical value.")

        scale = np.maximum(
            np.max(np.abs(self.char_poly_coeffs), axis=0), np.abs(analytical_coeffs)
        )
        scale = np.where(scale == 0, 1.0, scale)
        observations = self.char_poly_coeffs / scale
        target = analytical_coeffs / scale
        mean_coeffs = np.mean(observations, axis=0)
        std_coeffs = np.std(observations, axis=0, ddof=1)
        sem = std_coeffs / np.sqrt(self.samples)
        with np.errstate(over="ignore"):
            tolerance = atol / scale + rtol * np.abs(target)
        return bool(np.all(np.abs(mean_coeffs - target) <= critical * sem + tolerance))

    def plot(self, show: bool = True) -> Any:
        """
        Plots empirical eigenvalue distribution alongside theoretical polynomial roots.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn(
                "matplotlib is required for plotting. Skipping plot.", stacklevel=2
            )
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        flat_eigs = self.eigenvalues.flatten()

        ax.hist(
            flat_eigs,
            bins=50,
            density=True,
            alpha=0.6,
            color="#1f77b4",
            edgecolor="none",
            label="Empirical Eigenvalues",
        )

        roots = self.analytical_poly.evaluate_roots_float64()
        ax.vlines(
            roots,
            ymin=0,
            ymax=ax.get_ylim()[1] * 0.1,
            colors="#d62728",
            linewidth=1.5,
            label="Analytical Roots",
        )

        ax.set_title(
            f"Empirical Spectral Density vs. Analytical Roots (d={self.d}, samples={self.samples})"
        )
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        if show:
            plt.show()
        return fig
