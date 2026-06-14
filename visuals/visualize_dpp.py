import os
import math
import numpy as np
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import flint
import sympy as sp

from finitefree import (
    OrthogonalPolynomialKernel,
    hermite_polynomial,
    gap_probability_continuous,
    sample_discrete,
    FiniteRTransform,
    RealRootedPolynomial,
    gue_expected_poly,
)
from finitefree.utils.conversion import flint_to_float

os.makedirs("visuals/assets", exist_ok=True)


def get_hermite_kernel(d: int) -> OrthogonalPolynomialKernel:
    # Use physicist's Hermite polynomials (monic versions returned by hermite_polynomial with physicist=True)
    polys = [hermite_polynomial(k, physicist=True) for k in range(d + 1)]
    norms = [flint.fmpq(int(math.factorial(k)), int(2**k)) for k in range(d)]
    return OrthogonalPolynomialKernel(polys, norms)


def weight_func_hermite(x):
    # Normalized weight function for physicist's Hermite polynomials
    return np.exp(-x**2) / np.sqrt(np.pi)


def hermite_function_kernel_matrix(d: int, xs: np.ndarray) -> np.ndarray:
    """
    Computes the weighted GUE kernel K_d(x, y) * exp(-x^2/2 - y^2/2) / pi^1/4
    exactly and stably using the Hermite function recurrence.
    This avoids numerical overflow and underflow in float64.
    """
    M = len(xs)
    phi = np.zeros((d + 1, M))
    phi[0] = np.exp(-xs**2 / 2.0) / (np.pi ** 0.25)
    if d > 0:
        phi[1] = np.sqrt(2.0) * xs * phi[0]
    for k in range(1, d):
        phi[k+1] = np.sqrt(2.0 / (k + 1)) * xs * phi[k] - np.sqrt(float(k) / (k + 1)) * phi[k-1]
        
    # CD formula off-diagonal
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    numerator = np.outer(phi[d], phi[d-1]) - np.outer(phi[d-1], phi[d])
    diff = X - Y
    np.fill_diagonal(diff, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        K_mat = np.sqrt(d / 2.0) * numerator / diff
    
    # Diagonal: \phi_k' = \sqrt{2k} \phi_{k-1} - x \phi_k
    phi_d_prime = np.sqrt(2.0 * d) * phi[d-1] - xs * phi[d]
    phi_d_minus_prime = np.sqrt(2.0 * (d - 1)) * phi[d-2] - xs * phi[d-1] if d > 1 else -xs * phi[0]
    diag_vals = np.sqrt(d / 2.0) * (phi_d_prime * phi[d-1] - phi_d_minus_prime * phi[d])
    np.fill_diagonal(K_mat, diag_vals)
    return K_mat


def project_kernel_to_rank(K_weighted: np.ndarray, d: int) -> np.ndarray:
    """
    Projects the discretized kernel matrix K_weighted to a true rank d projection matrix.
    This ensures eigenvalues are exactly 1.0 (d times) and 0.0 (rest), which guarantees
    HKPV samples exactly d points and matches the expected polynomial.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(K_weighted)
    idx = np.argsort(eigenvalues)[::-1][:d]
    V_d = eigenvectors[:, idx]
    return V_d @ V_d.T


class PrecomputedDiscreteKernel:
    def __init__(self, K):
        self._K = K
    def __call__(self, i, j):
        return self._K[int(i)][int(j)]
    def matrix(self, xs):
        indices = [int(x) for x in xs]
        return self._K[np.ix_(indices, indices)].tolist()


def animate_asymptotic_kernel_scaling():
    from PIL import Image
    print("Generating Asymptotic Kernel Scaling Limits animation...")
    degrees = [10, 20, 30, 45, 60, 80, 100, 130, 160, 200]
    frame_images = []
    
    os.makedirs("visuals/assets/temp_kernel_frames", exist_ok=True)
    
    # Bulk points and theoretical limit
    y_fixed = 0.0
    x_vals = np.linspace(-3.0, 3.0, 300)
    sine_limit = np.sinc(x_vals)
    
    # Edge points and theoretical limit
    y_edge_fixed = 0.0
    x_edge_vals = np.linspace(-4.0, 2.0, 300)
    ai_y, aip_y, _, _ = scipy.special.airy(y_edge_fixed)
    airy_limit = []
    for x in x_edge_vals:
        if np.isclose(x, y_edge_fixed):
            ai_x, aip_x, _, _ = scipy.special.airy(x)
            val = aip_x**2 - x * ai_x**2
        else:
            ai_x, aip_x, _, _ = scipy.special.airy(x)
            val = (ai_x * aip_y - aip_x * ai_y) / (x - y_edge_fixed)
        airy_limit.append(val)
        
    for idx, d in enumerate(degrees):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Bulk
        rho_0 = np.sqrt(2.0 * d) / np.pi
        pts = np.concatenate([x_vals / rho_0, [y_fixed / rho_0]])
        K_mat = hermite_function_kernel_matrix(d, pts)
        k_bulk = K_mat[:-1, -1] / rho_0
        
        ax1.plot(x_vals, k_bulk, color='#1f77b4', lw=2.0, label=f"Finite Kernel (d={d})")
        ax1.plot(x_vals, sine_limit, '--', color='black', lw=2.0, label="Sine Kernel Limit")
        ax1.set_title("Bulk Universality (Sine Kernel Limit)", fontsize=11, fontweight='bold')
        ax1.set_xlabel(r"Rescaled Distance $\pi(x - y)$")
        ax1.set_ylabel("Kernel Value")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")
        
        # 2. Edge
        edge = np.sqrt(2.0 * d)
        scale = 1.0 / (np.sqrt(2.0) * (d ** (1.0/6.0)))
        pts_edge = np.concatenate([edge + x_edge_vals * scale, [edge + y_edge_fixed * scale]])
        K_mat_edge = hermite_function_kernel_matrix(d, pts_edge)
        k_edge = K_mat_edge[:-1, -1] * scale
        
        ax2.plot(x_edge_vals, k_edge, color='#2ca02c', lw=2.0, label=f"Finite Kernel (d={d})")
        ax2.plot(x_edge_vals, airy_limit, '--', color='black', lw=2.0, label="Airy Kernel Limit")
        ax2.set_title("Edge Universality (Airy Kernel Limit)", fontsize=11, fontweight='bold')
        ax2.set_xlabel(r"Rescaled Edge Distance")
        ax2.set_ylabel("Kernel Value")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right")
        
        plt.suptitle(f"Kernel Scaling Limits Convergence (d={d})", fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        frame_path = f"visuals/assets/temp_kernel_frames/frame_{idx:03d}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close()
        
        frame_images.append(Image.open(frame_path))
        
    gif_path = "visuals/assets/asymptotic_kernel_scaling.gif"
    if frame_images:
        frame_images[0].save(
            gif_path,
            save_all=True,
            append_images=frame_images[1:],
            duration=300,
            loop=0
        )
    print(f"Saved kernel scaling animation to {gif_path}")
    
    for img in frame_images:
        img.close()
        
    # Cleanup
    for idx in range(len(degrees)):
        try:
            os.remove(f"visuals/assets/temp_kernel_frames/frame_{idx:03d}.png")
        except OSError:
            pass
    try:
        os.rmdir("visuals/assets/temp_kernel_frames")
    except OSError:
        pass
    print("Saved animation to visuals/assets/asymptotic_kernel_scaling.gif")


def gap_probability_continuous_vectorized(d: int, a: float, b: float, n_points: int = 35) -> float:
    """
    Approximates the continuous Fredholm determinant gap probability over [a, b] using Nyström discretization
    natively with the vectorized float64 hermite_function_kernel_matrix.
    """
    pts, w = scipy.special.roots_legendre(n_points)
    # Map points and weights from [-1, 1] to [a, b]
    pts_mapped = 0.5 * (b - a) * pts + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * w

    K_mat = hermite_function_kernel_matrix(d, pts_mapped)
    
    # K_mat already contains the weight factors exp(-x^2/2 - y^2/2) / pi^0.25
    # So we only need to apply the quadrature weights
    w_sqrt = np.sqrt(w_mapped)
    D = w_sqrt[:, None] * K_mat * w_sqrt[None, :]

    matrix = np.eye(n_points) - D
    return float(np.linalg.det(matrix))


def visualize_tracy_widom_convergence():
    print("Generating Tracy-Widom Convergence...")
    
    d = 100
    M = 400
    edge = np.sqrt(2.0 * d)
    scale = (2.0 ** -0.5) * (d ** (-1.0/6.0))
    state_space = np.linspace(-15.0, 16.5, M)
    delta_x = (state_space[-1] - state_space[0]) / (M - 1)
    
    # Compute and project to rank d
    with np.errstate(divide='ignore', invalid='ignore'):
        K_mat_weighted = hermite_function_kernel_matrix(d, state_space) * delta_x
    K_proj = project_kernel_to_rank(K_mat_weighted, d)
    state_space_indices = list(range(M))
    
    num_samples = 400
    max_eigenvalues = []
    print(f"Sampling {num_samples} DPP configurations via HKPV...")
    for _ in range(num_samples):
        conf = sample_discrete(K_proj, state_space_indices)
        if len(conf) > 0:
            vals = [state_space[idx] for idx in conf]
            max_eigenvalues.append(np.max(vals))
            
    max_eigenvalues = np.array(max_eigenvalues)
    rescaled_max = (max_eigenvalues - edge) / scale
    
    # Analytical Fredholm Determinant Tracy-Widom Beta=2
    s_grid = np.linspace(-4.0, 2.5, 30)
    analytical_probs = []
    for s in s_grid:
        s_cont = edge + s * scale
        prob = gap_probability_continuous_vectorized(d, s_cont, max(s_cont + 1.0, 18.0), n_points=35)
        analytical_probs.append(prob)
        
    sorted_rescaled = np.sort(rescaled_max)
    empirical_cdf = np.arange(1, len(sorted_rescaled) + 1) / len(sorted_rescaled)
    
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_rescaled, empirical_cdf, label="Empirical CDF (HKPV Sampler)", color="#1f77b4", lw=2)
    plt.plot(s_grid, analytical_probs, 'o--', label="Analytical Fredholm Determinant (Continuous)", color="#d62728")
    
    plt.title(f"Tracy-Widom Convergence (GUE Edge, d={d})", fontsize=13, fontweight='bold')
    plt.xlabel("Rescaled Max Eigenvalue $s$")
    plt.ylabel(r"$P(\lambda_{\max} \leq s)$")
    plt.xlim(-4.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals/assets/tracy_widom_convergence.png", dpi=150)
    plt.close()
    print("Saved plot to visuals/assets/tracy_widom_convergence.png")


def visualize_level_repulsion():
    print("Generating Level Repulsion spacing distribution...")
    
    d = 16
    M = 600
    state_space = np.linspace(-5.5, 5.5, M)
    delta_x = 11.0 / (M - 1)
    
    K_mat_weighted = hermite_function_kernel_matrix(d, state_space) * delta_x
    K_proj = project_kernel_to_rank(K_mat_weighted, d)
    state_space_indices = list(range(M))
    
    unfold_grid = np.linspace(-5.6, 5.6, 2000)
    diag_density = np.diag(hermite_function_kernel_matrix(d, unfold_grid))
    cdf_vals = scipy.integrate.cumulative_trapezoid(diag_density, unfold_grid, initial=0.0)
    
    def GUE_CDF(x):
        return float(np.interp(x, unfold_grid, cdf_vals))
        
    num_samples = 4000
    spacings = []
    
    print(f"Sampling {num_samples} DPP configurations...")
    for _ in range(num_samples):
        conf = sample_discrete(K_proj, state_space_indices)
        dithered_indices = np.array(conf) + np.random.uniform(-0.5, 0.5, len(conf))
        vals = state_space[0] + dithered_indices * delta_x
        unfolded_vals = np.sort([GUE_CDF(v) for v in vals])
        middle_vals = unfolded_vals[4:d-4]
        diffs = np.diff(middle_vals)
        for diff in diffs:
            spacings.append(diff)
                    
    spacings = np.array(spacings)
    mean_spacing = np.mean(spacings)
    normalized_spacings = spacings / mean_spacing
    
    plt.figure(figsize=(8, 6))
    plt.hist(normalized_spacings, bins=50, density=True, alpha=0.6, color="#2ca02c", edgecolor="none", label="Empirical Spacings")
    
    s_grid = np.linspace(0.0, 3.0, 200)
    wigner_surmise = (32.0 / (np.pi ** 2)) * (s_grid ** 2) * np.exp(-4.0 * (s_grid ** 2) / np.pi)
    
    plt.plot(s_grid, wigner_surmise, '-', color="#d62728", lw=2.5, label="Wigner Surmise (GUE, \u03b2=2)")
    plt.title("Nearest-Neighbor Spacing Distribution (Bulk Repulsion)", fontsize=13, fontweight='bold')
    plt.xlabel("Normalized Spacing $s$")
    plt.ylabel("Probability Density")
    plt.xlim(0.0, 3.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals/assets/level_repulsion.png", dpi=150)
    plt.close()
    print("Saved plot to visuals/assets/level_repulsion.png")


if __name__ == "__main__":
    animate_asymptotic_kernel_scaling()
    visualize_tracy_widom_convergence()
    visualize_level_repulsion()
