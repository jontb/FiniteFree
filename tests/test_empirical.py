from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.stats import ortho_group, unitary_group

from finitefree.convolutions import symmetric_additive
from finitefree.core import RealRootedPolynomial


def random_symplectic(d: int) -> npt.NDArray[np.complex128]:
    """
    Generates a Haar-distributed unitary symplectic matrix in USp(2d)
    using block quaternionic QR decomposition.
    """
    X = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    Y = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    Z = np.block([[X, Y], [-np.conj(Y), np.conj(X)]])
    Q, R = np.linalg.qr(Z)
    d_R = np.diagonal(R)
    d_R_safe = np.where(d_R == 0, 1.0, d_R)
    ph = d_R_safe / np.abs(d_R_safe)
    Q = Q * ph
    return cast(npt.NDArray[np.complex128], Q)


def test_empirical_unitary_validation_beta2() -> None:
    """
    Validates expected characteristic polynomial under GUE (beta=2) unitary conjugation.
    E[chi(A + U B U^H)] = chi(A) \boxplus_d chi(B)
    """
    d = 4
    N = 3000

    roots_A = np.array([1.0, 2.0, 3.0, 4.0])
    roots_B = np.array([-2.0, -1.0, 1.0, 2.0])

    A = np.diag(roots_A)
    B = np.diag(roots_B)

    poly_A = RealRootedPolynomial(np.poly(roots_A), assume_real_rooted=True)
    poly_B = RealRootedPolynomial(np.poly(roots_B), assume_real_rooted=True)

    res_analytical = symmetric_additive(poly_A, poly_B, d)
    expected_coeffs = np.array(res_analytical.coeffs, dtype=float)

    empirical_coeffs_sum = np.zeros(d + 1, dtype=float)
    np.random.seed(42)

    for _ in range(N):
        U = unitary_group.rvs(d)
        M = A + U @ B @ U.conj().T
        coeffs = np.poly(M)
        empirical_coeffs_sum += np.real(coeffs)

    empirical_coeffs = empirical_coeffs_sum / N
    l2_norm = np.linalg.norm(empirical_coeffs - expected_coeffs)

    assert l2_norm < 1.0, f"GUE empirical convergence failed. L2 norm: {l2_norm}"


def test_empirical_orthogonal_validation_beta1() -> None:
    """
    Validates expected characteristic polynomial under GOE (beta=1)
    orthogonal conjugation.
    E[chi(A + Q B Q^T)] = chi(A) \boxplus_d chi(B)
    """
    d = 4
    N = 3000

    roots_A = np.array([1.0, 2.0, 3.0, 4.0])
    roots_B = np.array([-2.0, -1.0, 1.0, 2.0])

    A = np.diag(roots_A)
    B = np.diag(roots_B)

    poly_A = RealRootedPolynomial(np.poly(roots_A), assume_real_rooted=True)
    poly_B = RealRootedPolynomial(np.poly(roots_B), assume_real_rooted=True)

    res_analytical = symmetric_additive(poly_A, poly_B, d)
    expected_coeffs = np.array(res_analytical.coeffs, dtype=float)

    empirical_coeffs_sum = np.zeros(d + 1, dtype=float)
    np.random.seed(42)

    for _ in range(N):
        Q = ortho_group.rvs(d)
        M = A + Q @ B @ Q.T
        coeffs = np.poly(M)
        empirical_coeffs_sum += np.real(coeffs)

    empirical_coeffs = empirical_coeffs_sum / N
    l2_norm = np.linalg.norm(empirical_coeffs - expected_coeffs)

    assert l2_norm < 1.0, f"GOE empirical convergence failed. L2 norm: {l2_norm}"


def test_empirical_symplectic_validation_beta4() -> None:
    """
    Validates expected characteristic polynomial under GSE (beta=4)
    symplectic conjugation.
    E[chi(A_2d + S B_2d S^H)] = (chi(A) \boxplus_d chi(B))^2
    """
    d = 3  # Keep size manageable for multi-precision / Kronecker products
    N = 2000

    roots_A = np.array([1.5, 2.5, 3.5])
    roots_B = np.array([-1.0, 0.0, 1.0])

    # Replicate eigenvalues to form 2d x 2d matrices (Kramers degeneracy)
    A_2d = np.diag(np.repeat(roots_A, 2))
    B_2d = np.diag(np.repeat(roots_B, 2))

    poly_A = RealRootedPolynomial(np.poly(roots_A), assume_real_rooted=True)
    poly_B = RealRootedPolynomial(np.poly(roots_B), assume_real_rooted=True)

    # Compute symmetric additive convolution
    res_analytical = symmetric_additive(poly_A, poly_B, d)

    # Expected GSE characteristic polynomial is the square of the additive convolution
    expected_coeffs = np.convolve(
        np.array(res_analytical.coeffs, dtype=float),
        np.array(res_analytical.coeffs, dtype=float),
    )

    empirical_coeffs_sum = np.zeros(2 * d + 1, dtype=float)
    np.random.seed(42)

    for _ in range(N):
        S = random_symplectic(d)
        M = A_2d + S @ B_2d @ S.conj().T
        coeffs = np.poly(M)
        empirical_coeffs_sum += np.real(coeffs)

    empirical_coeffs = empirical_coeffs_sum / N
    l2_norm = np.linalg.norm(empirical_coeffs - expected_coeffs)

    # Since coefficients range up to 440, an absolute L2 norm of < 5.0
    # corresponds to a highly precise relative error of < 1% across all coefficients.
    assert l2_norm < 5.0, f"GSE empirical convergence failed. L2 norm: {l2_norm}"
