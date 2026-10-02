import math
import operator
from typing import Any, Callable, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from .core import RealRootedPolynomial
from .utils.roots import _gaussian_roots, _laguerre_roots


def sample_gue(d: int, scale: float = 1.0) -> Any:
    r"""
    Generates a sample GUE matrix (Hermitian, Gaussian entries)
    with diagonal and off-diagonal variance of $1/d$.
    """
    X = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2.0)
    H = (X + X.conj().T) / (np.sqrt(2.0) * np.sqrt(d))
    return H * scale


def sample_goe(d: int, scale: float = 1.0) -> Any:
    r"""
    Generates a sample GOE matrix (Symmetric, Gaussian entries)
    with diagonal variance $2/d$ and off-diagonal variance $1/d$.
    """
    X = np.random.randn(d, d)
    H = (X + X.T) / (np.sqrt(2.0) * np.sqrt(d))
    return H * scale


def sample_gse(d: int, scale: float = 1.0) -> Any:
    r"""
    Generates a sample GSE matrix of dimension $2d \times 2d$ (Self-dual, Gaussian entries)
    with Kramers degeneracy, scaled to match the $d$-dimensional expected characteristic polynomial.
    """
    X = (np.random.randn(2 * d, 2 * d) + 1j * np.random.randn(2 * d, 2 * d)) / np.sqrt(
        2.0
    )
    H = (X + X.conj().T) / 2.0
    J = np.zeros((2 * d, 2 * d))
    J[:d, d:] = np.eye(d)
    J[d:, :d] = -np.eye(d)
    H = (H + J @ H.conj() @ J.T) / 2.0
    return H * np.sqrt(2.0 / d) * scale


def sample_wishart(d: int, n: int, beta: int = 2, scale: float = 1.0) -> NDArray[Any]:
    r"""
    Generates a sample Wishart (LUE for $\beta=2$, LOE for $\beta=1$, LSE for $\beta=4$)
    matrix $W = X X^H / n$.
    """
    if beta == 1:
        X = np.random.randn(d, n)
        W = (X @ X.T) / n
    elif beta == 2:
        X = (np.random.randn(d, n) + 1j * np.random.randn(d, n)) / np.sqrt(2.0)
        W = (X @ X.conj().T) / n
    elif beta == 4:
        A = (np.random.randn(d, n) + 1j * np.random.randn(d, n)) / np.sqrt(2.0)
        B = (np.random.randn(d, n) + 1j * np.random.randn(d, n)) / np.sqrt(2.0)
        X = np.block([[A, B], [-np.conj(B), np.conj(A)]])
        W = (X @ X.conj().T) / (2 * n)
    else:
        raise ValueError("beta must be 1, 2, or 4")
    return W * scale


def sample_haar_unitary(d: int) -> Any:
    """Generates a Haar-distributed random unitary matrix."""
    X = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(X)
    d_r = np.diagonal(R)
    ph = d_r / np.abs(d_r)
    return Q * ph


def sample_haar_orthogonal(d: int) -> Any:
    """Generates a Haar-distributed random orthogonal matrix."""
    X = np.random.randn(d, d)
    Q, R = np.linalg.qr(X)
    d_r = np.diagonal(R)
    ph = d_r / np.abs(d_r)
    return Q * ph


def sample_haar_symplectic(d: int) -> NDArray[np.complex128]:
    r"""Generates a Haar-distributed random symplectic matrix in $\text{USp}(2d)$."""
    X = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    Y = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
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
    Computes the exact expected characteristic polynomial of a $d \times d$ Wishart matrix:
    $\mathbb{E}[\det(xI - W)] = n^{-d} d! (-1)^d L_d^{(n - d)}(n x)$
    using the generalized Laguerre polynomial from orthogonal.py.
    """
    import flint

    from .orthogonal import laguerre_polynomial

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
            M = np.asarray(generator(), dtype=np.complex128)
            if M.shape not in [(self.d, self.d), (2 * self.d, 2 * self.d)]:
                raise ValueError("Sample matrices must have shape (d, d) or (2d, 2d).")
            if not np.all(np.isfinite(M)):
                raise ValueError("Sample matrices must contain finite entries.")
            matrix_scale = np.max(np.abs(M), initial=0.0)
            if not np.isfinite(matrix_scale) or not np.allclose(
                M, M.conj().T, rtol=0, atol=1e-10 * matrix_scale
            ):
                raise ValueError("Sample matrices must be numerically Hermitian.")
            eigs = np.linalg.eigvalsh(M)
            if not np.all(np.isfinite(eigs)):
                raise ValueError("Sample eigenvalues must be finite.")
            if self.d and M.shape[0] == 2 * self.d:
                # A doubled-size sample is usable only with Kramers pairs.
                eigenvalue_scale = np.max(np.abs(eigs))
                if not np.allclose(
                    eigs[::2], eigs[1::2], rtol=0, atol=1e-10 * eigenvalue_scale
                ):
                    raise ValueError(
                        "Doubled-size samples must have paired eigenvalues."
                    )
                eigs = eigs[::2]

            with np.errstate(over="ignore", invalid="ignore"):
                coefficients = np.atleast_1d(np.poly(eigs))
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("Sample characteristic coefficients must be finite.")

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
