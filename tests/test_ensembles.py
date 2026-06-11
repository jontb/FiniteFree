import numpy as np

from finitefree import (
    EmpiricalComparison,
    gue_expected_poly,
    sample_goe,
    sample_gse,
    sample_gue,
    sample_haar_orthogonal,
    sample_haar_symplectic,
    sample_haar_unitary,
    sample_wishart,
    wishart_expected_poly,
)


def test_gue_sampler() -> None:
    d = 5
    H = sample_gue(d)
    assert H.shape == (d, d)
    # Check Hermitian property
    assert np.allclose(H, H.conj().T)
    # Check that diagonal is real
    assert np.allclose(np.imag(np.diag(H)), 0)


def test_goe_sampler() -> None:
    d = 5
    H = sample_goe(d)
    assert H.shape == (d, d)
    # Check Symmetric property
    assert np.allclose(H, H.T)
    # Check that entries are real
    assert np.isrealobj(H)


def test_gse_sampler() -> None:
    d = 4
    H = sample_gse(d)
    assert H.shape == (2 * d, 2 * d)
    # Check Hermitian property
    assert np.allclose(H, H.conj().T)

    # Check Symplectic self-dual structure: J H.conj() J^T = H
    J = np.zeros((2 * d, 2 * d))
    J[:d, d:] = np.eye(d)
    J[d:, :d] = -np.eye(d)
    assert np.allclose(H, J @ H.conj() @ J.T)


def test_wishart_sampler() -> None:
    d, n = 3, 5
    # beta=1 (LOE)
    W1 = sample_wishart(d, n, beta=1)
    assert W1.shape == (d, d)
    assert np.allclose(W1, W1.T)
    assert np.isrealobj(W1)

    # beta=2 (LUE)
    W2 = sample_wishart(d, n, beta=2)
    assert W2.shape == (d, d)
    assert np.allclose(W2, W2.conj().T)

    # beta=4 (LSE)
    W4 = sample_wishart(d, n, beta=4)
    assert W4.shape == (2 * d, 2 * d)
    assert np.allclose(W4, W4.conj().T)


def test_haar_samplers() -> None:
    d = 4
    # Unitary
    U = sample_haar_unitary(d)
    assert np.allclose(U @ U.conj().T, np.eye(d))

    # Orthogonal
    orth_mat = sample_haar_orthogonal(d)
    assert np.allclose(orth_mat @ orth_mat.T, np.eye(d))
    assert np.isrealobj(orth_mat)

    # Symplectic
    S = sample_haar_symplectic(d)
    assert np.allclose(S @ S.conj().T, np.eye(2 * d))
    J = np.zeros((2 * d, 2 * d))
    J[:d, d:] = np.eye(d)
    J[d:, :d] = -np.eye(d)
    # Symplectic matrix satisfies S^T J S = J
    assert np.allclose(S.T @ J @ S, J)


def test_empirical_gue_expected_poly() -> None:
    d = 4
    samples = 1000
    analytical_poly = gue_expected_poly(d)

    # Check degree
    assert analytical_poly.degree == d

    # Compare with empirical samples
    comp = EmpiricalComparison(
        analytical_poly=analytical_poly,
        samples=samples,
        generator=lambda: sample_gue(d)
    )

    # Coefficient mean validation (5-sigma confidence)
    assert comp.verify_coefficients()


def test_empirical_wishart_expected_poly() -> None:
    d = 3
    n = 6
    samples = 1000
    analytical_poly = wishart_expected_poly(d, n, beta=2)

    assert analytical_poly.degree == d

    comp = EmpiricalComparison(
        analytical_poly=analytical_poly,
        samples=samples,
        generator=lambda: sample_wishart(d, n, beta=2)
    )

    assert comp.verify_coefficients()


def test_empirical_unitary_validation_beta2() -> None:
    """
    Validates expected characteristic polynomial under GUE (beta=2) unitary conjugation.
    E[chi(A + U B U^H)] = chi(A) \boxplus_d chi(B)
    """
    from finitefree.convolutions import symmetric_additive
    from finitefree.core import RealRootedPolynomial
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
        U = sample_haar_unitary(d)
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
    from finitefree.convolutions import symmetric_additive
    from finitefree.core import RealRootedPolynomial
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
        Q = sample_haar_orthogonal(d)
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
    from finitefree.convolutions import symmetric_additive
    from finitefree.core import RealRootedPolynomial
    d = 3
    N = 2000

    roots_A = np.array([1.5, 2.5, 3.5])
    roots_B = np.array([-1.0, 0.0, 1.0])

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

    res_analytical = symmetric_additive(poly_A, poly_B, d)
    expected_coeffs = np.array(res_analytical.coeffs, dtype=float)

    all_coeffs = []
    np.random.seed(42)

    for _ in range(N):
        S = sample_haar_symplectic(d)
        M = A_2d + S @ B_2d @ S.conj().T

        roots = np.linalg.eigvalsh(M)
        sorted_roots = np.sort(roots)
        assert np.allclose(sorted_roots[::2], sorted_roots[1::2], atol=1e-5)

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
            assert diff <= 5 * sem[i], (
                f"GSE Coefficient {i} mismatch. Mean: {empirical_coeffs[i]}, "
                f"Expected: {expected_coeffs[i]}, SEM: {sem[i]}"
            )


def test_wishart_sampler_invalid_beta() -> None:
    import pytest
    with pytest.raises(ValueError, match="beta must be 1, 2, or 4"):
        sample_wishart(d=3, n=5, beta=3)

