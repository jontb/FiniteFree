import math
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .core import RealRootedPolynomial


def sample_gue(d: int, scale: float = 1.0) -> Any:
    """
    Generates a sample GUE matrix (Hermitian, Gaussian entries)
    with diagonal and off-diagonal variance of 1/d.
    """
    X = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2.0)
    H = (X + X.conj().T) / (np.sqrt(2.0) * np.sqrt(d))
    return H * scale


def sample_goe(d: int, scale: float = 1.0) -> Any:
    """
    Generates a sample GOE matrix (Symmetric, Gaussian entries)
    with diagonal variance 2/d and off-diagonal variance 1/d.
    """
    X = np.random.randn(d, d)
    H = (X + X.T) / (np.sqrt(2.0) * np.sqrt(d))
    return H * scale


def sample_gse(d: int, scale: float = 1.0) -> Any:
    """
    Generates a sample GSE matrix of dimension 2d x 2d (Self-dual, Gaussian entries)
    with Kramers degeneracy, scaled to match the d-dimensional expected characteristic polynomial.
    """
    X = (np.random.randn(2 * d, 2 * d) + 1j * np.random.randn(2 * d, 2 * d)) / np.sqrt(2.0)
    H = (X + X.conj().T) / 2.0
    J = np.zeros((2 * d, 2 * d))
    J[:d, d:] = np.eye(d)
    J[d:, :d] = -np.eye(d)
    H = (H + J @ H.conj() @ J.T) / 2.0
    return H * np.sqrt(2.0 / d) * scale


def sample_wishart(d: int, n: int, beta: int = 2, scale: float = 1.0) -> NDArray[Any]:
    """
    Generates a sample Wishart (LUE for beta=2, LOE for beta=1, LSE for beta=4)
    matrix W = X X^H / n.
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
    """Generates a Haar-distributed random symplectic matrix in USp(2d)."""
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
            W = Q[:, d:d+j]
            v = v - U @ (U.conj().T @ v) - W @ (W.conj().T @ v)
        u = v / np.linalg.norm(v)
        w = -J @ np.conj(u)
        Q[:, j] = u
        Q[:, j + d] = w
    return Q


def gue_expected_poly(d: int) -> RealRootedPolynomial:
    """
    Computes the exact expected characteristic polynomial of a d x d GUE matrix:
    E[det(xI - M)] = d^{-d/2} He_d(sqrt(d) x)
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
    return RealRootedPolynomial(new_poly, assume_real_rooted=True)


def wishart_expected_poly(d: int, n: int, beta: int = 2) -> RealRootedPolynomial:
    """
    Computes the exact expected characteristic polynomial of a d x d Wishart matrix:
    E[det(xI - W)] = n^{-d} d! (-1)^d L_d^{(n - d)}(n x)
    using the generalized Laguerre polynomial from orthogonal.py.
    """
    import flint

    from .orthogonal import laguerre_polynomial

    lag = laguerre_polynomial(d, n - d)
    f_poly = lag._fmpq_poly
    coeffs_list = list(f_poly)
    scaled_coeffs = []
    factor_base = flint.fmpq(math.factorial(d) * (-1)**d)

    for i, coeff in enumerate(coeffs_list):
        if coeff == 0:
            scaled_coeffs.append(flint.fmpq(0))
        else:
            power = i - d
            factor = factor_base * (flint.fmpq(n) ** power)
            scaled_coeffs.append(coeff * factor)

    new_poly = flint.fmpq_poly(scaled_coeffs)
    return RealRootedPolynomial(new_poly, assume_real_rooted=True)


class EmpiricalComparison:
    """
    Compares analytical Expected Characteristic Polynomials against matrix simulations.
    """
    def __init__(
        self,
        analytical_poly: RealRootedPolynomial,
        samples: int,
        generator: Callable[[], NDArray[Any]],
    ) -> None:
        self.analytical_poly = analytical_poly
        self.samples = samples
        self.generator = generator
        self.d = analytical_poly.degree

        eigs_list = []
        coeffs_list = []
        for _ in range(samples):
            M = generator()
            if M.shape[0] == 2 * self.d:
                # GSE Kramers degeneracy: extract unique eigenvalues
                eigs = np.linalg.eigvalsh(M)
                eigs = np.sort(eigs)[::2]
            else:
                eigs = np.linalg.eigvalsh(M)

            eigs_list.append(eigs)
            coeffs_list.append(np.poly(eigs))

        self.eigenvalues = np.array(eigs_list)
        self.char_poly_coeffs = np.array(coeffs_list)

    def verify_coefficients(self, alpha: float = 0.05) -> bool:
        """
        Validates empirical coefficients against theoretical ones using
        a strict 5-sigma statistical confidence check.
        """
        analytical_coeffs = np.array(self.analytical_poly.coeffs, dtype=float)
        mean_coeffs = np.mean(self.char_poly_coeffs, axis=0)
        std_coeffs = np.std(self.char_poly_coeffs, axis=0, ddof=1)
        sem = std_coeffs / np.sqrt(self.samples)

        for i in range(len(analytical_coeffs)):
            if sem[i] > 1e-10:
                diff = np.abs(mean_coeffs[i] - analytical_coeffs[i])
                if diff > 5 * sem[i]:
                    return False
        return True

    def plot(self, show: bool = True) -> Any:
        """
        Plots empirical eigenvalue distribution alongside theoretical polynomial roots.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn("matplotlib is required for plotting. Skipping plot.", stacklevel=2)
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        flat_eigs = self.eigenvalues.flatten()

        ax.hist(flat_eigs, bins=50, density=True, alpha=0.6, color="#1f77b4", edgecolor="none", label="Empirical Eigenvalues")

        roots = self.analytical_poly.evaluate_roots_float64()
        ax.vlines(roots, ymin=0, ymax=ax.get_ylim()[1] * 0.1, colors="#d62728", linewidth=1.5, label="Analytical Roots")

        ax.set_title(f"Empirical Spectral Density vs. Analytical Roots (d={self.d}, samples={self.samples})")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        if show:
            plt.show()
        return fig
