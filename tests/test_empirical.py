from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.stats import ortho_group, unitary_group

from finitefree.convolutions import symmetric_additive
from finitefree.core import RealRootedPolynomial


def random_symplectic(d: int) -> npt.NDArray[np.complex128]:
    """
    Generates a Haar-distributed unitary symplectic matrix in USp(2d)
    using quaternionic Gram-Schmidt.
    """
    X = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    Y = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    Z = np.block([[X, Y], [-np.conj(Y), np.conj(X)]])
    U_cols: list[tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]] = []
    J = np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])
    for j in range(d):
        v = Z[:, j]
        for u, w in U_cols:
            v = v - np.vdot(u, v) * u - np.vdot(w, v) * w
        u = v / np.linalg.norm(v)
        w = -J @ np.conj(u)
        U_cols.append((u, w))
    Q = np.zeros((2 * d, 2 * d), dtype=complex)
    for j in range(d):
        Q[:, j] = U_cols[j][0]
        Q[:, j + d] = U_cols[j][1]
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

    all_coeffs = []
    np.random.seed(42)

    for _ in range(N):
        U = unitary_group.rvs(d)
        M = A + U @ B @ U.conj().T
        coeffs = np.poly(M)
        all_coeffs.append(np.real(coeffs))

    all_coeffs_arr = np.array(all_coeffs)
    empirical_coeffs = np.mean(all_coeffs_arr, axis=0)
    empirical_std = np.std(all_coeffs_arr, axis=0, ddof=1)
    sem = empirical_std / np.sqrt(N)

    for i in range(len(expected_coeffs)):
        if sem[i] > 1e-10:
            diff = np.abs(empirical_coeffs[i] - expected_coeffs[i])
            assert diff <= 5 * sem[i], (
                f"GUE Coefficient {i} mismatch. Mean: {empirical_coeffs[i]}, "
                f"Expected: {expected_coeffs[i]}, SEM: {sem[i]}"
            )


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

    all_coeffs = []
    np.random.seed(42)

    for _ in range(N):
        Q = ortho_group.rvs(d)
        M = A + Q @ B @ Q.T
        coeffs = np.poly(M)
        all_coeffs.append(np.real(coeffs))

    all_coeffs_arr = np.array(all_coeffs)
    empirical_coeffs = np.mean(all_coeffs_arr, axis=0)
    empirical_std = np.std(all_coeffs_arr, axis=0, ddof=1)
    sem = empirical_std / np.sqrt(N)

    for i in range(len(expected_coeffs)):
        if sem[i] > 1e-10:
            diff = np.abs(empirical_coeffs[i] - expected_coeffs[i])
            assert diff <= 5 * sem[i], (
                f"GOE Coefficient {i} mismatch. Mean: {empirical_coeffs[i]}, "
                f"Expected: {expected_coeffs[i]}, SEM: {sem[i]}"
            )


def test_empirical_symplectic_validation_beta4() -> None:
    """
    Validates expected characteristic polynomial under GSE (beta=4)
    symplectic conjugation.
    E[chi_unique(A_2d + S B_2d S^H)] = chi(A) \boxplus_d chi(B)
    """
    d = 3  # Keep size manageable for multi-precision / Kronecker products
    N = 2000

    roots_A = np.array([1.5, 2.5, 3.5])
    roots_B = np.array([-1.0, 0.0, 1.0])

    # Replicate eigenvalues to form 2d x 2d matrices (Kramers degeneracy)
    # using diagonal block matrices to preserve quaternionic self-dual structure.
    A_2d = np.block(
        [
            [np.diag(roots_A), np.zeros((d, d))],
            [np.zeros((d, d)), np.diag(roots_A)],
        ]
    )
    B_2d = np.block(
        [
            [np.diag(roots_B), np.zeros((d, d))],
            [np.zeros((d, d)), np.diag(roots_B)],
        ]
    )

    poly_A = RealRootedPolynomial(np.poly(roots_A), assume_real_rooted=True)
    poly_B = RealRootedPolynomial(np.poly(roots_B), assume_real_rooted=True)

    # Compute symmetric additive convolution
    res_analytical = symmetric_additive(poly_A, poly_B, d)
    expected_coeffs = np.array(res_analytical.coeffs, dtype=float)

    all_coeffs = []
    np.random.seed(42)

    for _ in range(N):
        S = random_symplectic(d)
        M = A_2d + S @ B_2d @ S.conj().T

        # Verify Kramers degeneracy of the eigenvalues
        roots = np.linalg.eigvalsh(M)
        sorted_roots = np.sort(roots)
        assert np.allclose(sorted_roots[::2], sorted_roots[1::2], atol=1e-5)

        # Extract unique eigenvalues and construct unique characteristic polynomial
        u_roots = sorted_roots[::2]
        coeffs = np.poly(u_roots)
        all_coeffs.append(np.real(coeffs))

    all_coeffs_arr = np.array(all_coeffs)
    empirical_coeffs = np.mean(all_coeffs_arr, axis=0)
    empirical_std = np.std(all_coeffs_arr, axis=0, ddof=1)
    sem = empirical_std / np.sqrt(N)

    for i in range(len(expected_coeffs)):
        if sem[i] > 1e-10:
            diff = np.abs(empirical_coeffs[i] - expected_coeffs[i])
            # Strict 5-sigma statistical bound without arbitrary bias tolerances
            assert diff <= 5 * sem[i], (
                f"GSE Coefficient {i} mismatch. Mean: {empirical_coeffs[i]}, "
                f"Expected: {expected_coeffs[i]}, SEM: {sem[i]}"
            )
