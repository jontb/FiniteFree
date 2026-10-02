# FiniteFree: Finite Free Probability Library

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

FiniteFree is a precision-focused Python framework designed to operationalize the discrete calculus of finite free probability. It extends classical free probability limits to finite-dimensional polynomial representations (characteristic polynomials), mapping linear deterministic operators against polynomial distributions.

To prevent numerical floating-point drift and avoid the computational complexity of high-order runtime differential operators, FiniteFree implements exact finite free convolutions using discrete algebraic representations, exact generating function recurrences, and arbitrary-precision integer and rational scaling.

## Visual Showcase

The following showcase provides a visual representation of FiniteFree's features, organized by topic:

??? "1. Limiting Distributions of Free Convolutions"

    Animates the convergence of root distributions of expectation polynomials towards their theoretical free probability limits as the dimension/degree $d$ scales up to 300:

    - **Wigner Semicircle Convergence**: GUE expectation polynomials additive convolution $He_d \boxplus_d He_d$ roots converging to the semicircle distribution.
    - **Marchenko-Pastur Convergence**: Laguerre/Wishart expectation polynomials roots converging to the Marchenko-Pastur law.
    - **Free Jacobi Arcsine Convergence**: Jacobi/Legendre polynomials roots converging to the classical Arcsine limit.
    - **Free Log-Normal Convergence**: Multi-precision compound Wishart multiplicative convolutions roots converging to the free Log-normal distribution.

    | Wigner Semicircle Convergence ($He_{d} \boxplus_{d} He_{d}$ as $d \to 300$) | Marchenko-Pastur Convergence (Laguerre as $d \to 300$) |
    | :---: | :---: |
    | ![Wigner Semicircle Convergence](visuals/assets/wigner_semicircle_convergence.gif) | ![Marchenko-Pastur Convergence](visuals/assets/marchenko_pastur_convergence.gif) |

    | Free Jacobi Arcsine Convergence (Legendre as $d \to 300$) | Free Log-Normal Convergence (Wishart as $m=d \to 300$) |
    | :---: | :---: |
    | ![Free Jacobi Arcsine Convergence](visuals/assets/free_jacobi_arcsine_convergence.gif) | ![Free Log-Normal Convergence](visuals/assets/free_lognormal_convergence.gif) |


??? "2. Determinantal Point Processes & Spectral Universality"

    Showcases GUE spectral universality and eigenvalue spacing statistics:

    - **CD Bulk Scaling Limit**: Christoffel-Darboux kernel bulk scaling limit convergence to the infinite Sine kernel.
    - **CD Edge Scaling Limit**: Christoffel-Darboux kernel edge scaling limit convergence to the infinite Airy kernel.
    - **Tracy-Widom Edge Convergence**: Empirical max eigenvalue CDF from HKPV samples converging to the Tracy-Widom (Fredholm determinant) distribution.
    - **Nearest-Neighbor Level Repulsion**: GUE bulk eigenvalue spacing statistics matching the analytical Wigner Surmise.

    | CD Bulk Scaling Limit (Sine Kernel) | CD Edge Scaling Limit (Airy Kernel) |
    | :---: | :---: |
    | ![Asymptotic Bulk Scaling](visuals/assets/asymptotic_kernel_bulk.gif) | ![Asymptotic Edge Scaling](visuals/assets/asymptotic_kernel_edge.gif) |

    | Tracy-Widom Edge Convergence ($d=100$) | Nearest-Neighbor Level Repulsion (Bulk Spacings vs Wigner Surmise) |
    | :---: | :---: |
    | ![Tracy-Widom Convergence](visuals/assets/tracy_widom_convergence.png) | ![Level Repulsion Spacing](visuals/assets/level_repulsion.png) |


??? "3. Algebraic Properties"

    Illustrates exact algebraic properties, root interlacing constraints, and geometric distributions of FiniteFree polynomials:

    - **Root Interlacing Preservation**: Verifies that root interlacing relations ($p \prec q$) are strictly preserved under symmetric additive convolution ($p \boxplus_6 r \prec q \boxplus_6 r$) and multiplicative convolution ($p \boxtimes_6 r \prec q \boxtimes_6 r$).
    - **Multivariate Pencil Topography**: Visualizes the log-determinant topography and exact eigenvalue boundaries of random symmetric matrix pencils.
    - **Unitary Eigenvalue Repulsion**: Visualizes roots evolving on the complex unit circle $\mathbb{T}$ under unitary Hermite polynomial flows.

    | Original Roots ($p \prec q$) | Additive Convolution ($p \boxplus_6 r \prec q \boxplus_6 r$) | Multiplicative Convolution ($p \boxtimes_6 r \prec q \boxtimes_6 r$) |
    | :---: | :---: | :---: |
    | ![Original Roots](visuals/assets/root_interlacing_original.png) | ![Additive Interlacing](visuals/assets/root_interlacing_additive.png) | ![Multiplicative Interlacing](visuals/assets/root_interlacing_multiplicative.png) |

    | Hyperbolic Matrix Pencil Eigencones & Topography | Unitary Hermite Eigenvalue Repulsion Trajectories |
    | :---: | :---: |
    | ![Hyperbolic Cones](visuals/assets/hyperbolic_cones.png) | ![Unitary Trajectories](visuals/assets/unitary_trajectories.gif) |


??? "4. Asymptotic Convergence of Finite Free Transforms"

    Visualizes the convergence of exact finite free transforms to their continuous free probability limits as the dimension $d$ scales up to 300:

    - **T-Transform Step Convergence**: Fujie-Ueda Finite T-Transform step function convergence to the continuous free Marchenko-Pastur (MP) limit $T(t)=1+t$.
    - **Asymptotic Decay of Cumulants**: Linearizing additive convolution, showing exact high-order finite free cumulants $\kappa_k(d)$ decaying to zero, leaving only $\kappa_2 \to 1$ (the Wigner semicircle limit).
    - **S-Transform Multiplicativity**: S-transforms of Wishart expectation polynomials, demonstrating exact multiplicativity $S_{p \boxtimes_d q}^{(d)} = S_p^{(d)} S_q^{(d)}$ at all discrete nodes and convergence to the free S-transform limit.
    - **Cauchy Transform Convergence**: Domain coloring of the Finite Cauchy Transform $G_d(z) = \frac{1}{d} \frac{p'_d(z)}{p_d(z)}$, showing the roots (poles) and derivative roots (zeros) coalescing into a continuous branch cut $[-2, 2]$ of the semicircle law.

    | Fujie-Ueda T-Transform Convergence | Asymptotic Decay of Cumulants |
    | :---: | :---: |
    | ![T-Transform Convergence](visuals/assets/t_transform_convergence.gif) | ![Cumulant Decay](visuals/assets/cumulant_decay.gif) |

    | S-Transform Convergence | Cauchy Transform Convergence |
    | :---: | :---: |
    | ![S-Transform Convergence](visuals/assets/s_transform_convergence.gif) | ![Cauchy Transform Convergence](visuals/assets/cauchy_domain_coloring.gif) |



## Features

??? "Core Polynomial Operations & Domain Verification"

    - **Certified Real-Rooted Polynomial Validation**: Verified lazily with exact rational Sturm sequences for small squarefree factors and certified Arb isolation for larger factors. Exact Sturm remains a fallback through degree 30 when Arb cannot obtain a certificate.
    - **Unitary Circle Geometries ($\mathbb{T}$)**: Implements `UnitaryPolynomial` structures for polynomials with roots strictly on the complex unit circle, bypassing real-line Sturm sequence constraints and isolating angular arguments via complex eigensolvers (e.g., `unitary_hermite_polynomial`).
    - **Lazy Geometric Domain Properties**: After certifying real-rootedness, `has_non_negative_roots` and `has_strictly_positive_roots` use an $O(d)$ coefficient sign check to enforce operator domains. Sign alternation alone cannot certify real-rootedness.
    - **Basic Polynomial Transformations**: Supports exact algebraic transformations including variable dilation (`dilation`), variable shift (`shift`), positive integer root powers (`power`), root-reciprocal reversing (`reversed_polynomial`), derivative (`derivative`), projection (`projection`), fractional additive convolution power (`additive_power`), and the Fujie-Ueda limiting polynomial $\Phi_d$ (`phi_d`). Noninteger root powers use numerical root isolation and reconstruction.
    - **Direct Projections**: `projection(j)` computes the monic derivative of order `d-j` from the leading `j+1` coefficients, avoiding intermediate derivative objects. Known Hermite inputs retain their variance, center and numerical root recurrence. Projection to the original degree returns the same object. See the [API reference](api.md#polynomial-projections) for normalization and benchmark scope.
    - **High-Degree Scaling & Root Reconstruction**:
      - **Divide-and-Conquer Polynomial Synthesis**: `RealRootedPolynomial.from_roots(roots)` executes a binary splitting tree algorithm operating in $O(d \log^2 d)$ time for high-speed, exact polynomial synthesis.
    - **Optimized Memory Pool Management**: Prevents FLINT/GMP memory fragmentation at extreme degrees ($d \ge 1000$) through a conditional, threshold-based garbage collection registry inside `PrecisionContext`.


??? "Orthogonal Polynomial Families"

    - **Jacobi Polynomials** (`jacobi_polynomial`): Exact $O(n^2)$ recurrence construction of $P_n^{(\alpha, \beta)}(x)$ using `flint.fmpq_poly` in exact rational arithmetic.
    - **Hahn Polynomials** (`hahn_polynomial`): Exact $O(n)$ recurrence construction of $Q_n(x; \alpha, \beta, N)$ using sequential running products to prevent redundant Pochhammer factorial operations.
    - **Jack Polynomials** (`jack_polynomial`): High-performance variables recursion computing symmetric Jack polynomials $J_\lambda^{(\alpha)}(x_1, \dots, x_m)$ strictly over sparse monomial exponent dictionaries, completely bypassing SymPy's slow symbolic expansion engine.
    - **Chebyshev & Legendre Polynomials** (`chebyshev_t_polynomial`, `chebyshev_u_polynomial`, `legendre_polynomial`): Exact recurrence constructions of $T_n(x)$, $U_n(x)$, and $P_n(x)$ over $\mathbb{Q}$ via `flint.fmpq_poly`.


??? "Finite Free Convolutions"

    - **Symmetric Additive ($\boxplus_d$)**: Exact analytical evaluation mapping polynomial convolutions against the symmetric finite combinatorial variance.
    - **Asymmetric Additive ($\uplus_d$)**: Validated combinatorial operator supporting mixed rank geometry via fractional squared weights.
    - **Multiplicative ($\boxtimes_d$)**: Exact discrete multiplicative root transformations via scaled Hadamard projections.


??? "Analytical Finite Transforms"

    - **Finite Cauchy Transform** ($G_p^{(d)}$)
    - **Finite S-Transform** ($S_p^{(d)}$): Discrete normalized evaluations bypassing non-linear mapping.
    - **Finite R-Transform** ($R_p^{(d)}$): Computes finite free cumulants ($\kappa_n^{(d)}$) via an exact $O(n^2)$ recursive generating function sequence map over $\mathbb{Q}$, bypassing exponential partition lattice enumeration while verifying exact additivity $\kappa_n^{(d)}(p \boxplus_d q) = \kappa_n^{(d)}(p) + \kappa_n^{(d)}(q)$.
    - **Finite T-Transform** ($T_p^{(d)}$): Step function mapping the right-continuous inverse to the Fujie-Ueda limit $\Phi_d$, evaluated in $O(d)$ algebraically using coefficient sign-alternation validation.
    - **Symmetric Finite S-Transform** ($\tilde{S}_p^{(2d)}$): Evaluates the discrete S-Transform over symmetric domains via exact root-squaring maps ($\mathbf{Sq}(p)$), bypassing zero-valued odd coefficients.


??? "Multivariate Hyperbolic Geometry & Matrix Pencils"

    - **`MultivariatePolynomial`**: Homogeneous multivariate polynomials with exact directional derivatives, mixed partials, and homogeneous multinomial normalization.
    - **Compiled Sparse Evaluations**: Features `to_fmpq_mpoly()` for $O(1)$ evaluation and substitution using compiled C-level sparse representations inside the FLINT library.
    - **Jacobi SLP & Reverse AD**: Straightline programs evaluating determinant gradients and Hessians via C-level FLINT evaluations and exact, linear-time Reverse-Mode Automatic Differentiation (AD) over $\mathbb{Q}$ (completely bypassing symbolic expansion).
    - **Product-Grid Modular Interpolation (CRT)**: Evaluates exact determinant polynomials and pencil characteristic polynomials (bypassing symbolic expansion bottlenecks via exact rational interpolation over $\mathbb{Q}[t]$) using C-level `nmod_mat` solvers and the Chinese Remainder Theorem.
    - **Zippel Sparse Interpolation**: Deploys Zippel's probabilistic algorithm over $\mathbb{F}_p$ for sparse determinant evaluations (`from_symmetric_matrix_pencil_sparse`), bounding interpolation complexity to the target monomial count rather than the maximum total-degree combinatorial grid.
    - **Multiplicative Pencils & Diagonal Specialization**: Extends exact matrix pencil geometries to generalized asymmetric forms (`MultiplicativeMatrixPencil`) and computes univariate characteristic projections via optimized 1D Chinese Remainder Theorem loops.
    - **LMI Cone Verification**: Positive definiteness checks ($A(e) \succ 0$) to verify hyperbolic cones.


??? "Stable Numerical Root Evaluation"

    - **`to_numpy_poly1d()`**: Safe rational coefficient `float64` casting, utilizing arbitrary-precision `decimal.Decimal` fallbacks to prevent `OverflowError` on coefficients with extreme magnitude ratios.
    - **`evaluate_roots_float64(exact=False)`**: Uses scaled symmetric tridiagonal eigenvalues for Hermite, Laguerre, Jacobi, Legendre and Chebyshev polynomials in their orthogonality domains, including GUE/Wishart expectations, affine transforms and proven Hermite additive convolutions. It avoids forming an ill-conditioned monomial companion matrix for these families.
    - **`evaluate_roots_float64(exact=True)`**: Requests the Arb isolation path by default. A high-precision request after a numerical call bypasses the approximate cache. Results are returned as `float64`; if isolation fails, the general numerical fallbacks can still be used. See the [API reference](api.md#root-evaluation-and-conditioning) for dispatch, precision and conditioning limits.


??? "Random Matrix Ensembles (`finitefree.ensembles`)"

    - **Matrix Samplers**: Fast generation of invariant random matrices for GOE ($O(d)$, $\beta=1$), GUE ($U(d)$, $\beta=2$), and GSE ($USp(2d)$, $\beta=4$).
    - **Empirical Validations**: Computes theoretical expected characteristic polynomials $\mathbb{E}[\det(xI - M)]$ matching explicit orthogonal sequences.
    - **`EmpiricalComparison`**: Compares all sampled characteristic coefficients, including deterministic ones, with Bonferroni-adjusted two-sided Student-t bands. `alpha` controls the nominal family significance level; coefficient scaling prevents variance overflow/underflow. Samples must be finite Hermitian matrices of the matching size, with verified eigenvalue pairs for doubled-size representations. See the [API reference](api.md#empirical-coefficient-comparison) for sample assumptions and numerical tolerances.


??? "Determinantal Point Processes (`finitefree.dpp`)"

    - **Kernels**: Construct exact discrete DPP kernels and orthogonal polynomial kernels (e.g., Hermite/Laguerre) via the Christoffel-Darboux formula. Exact arguments retain their distinct values. Nearby distinct floating points use the finite basis sum to avoid a cancelled quotient; diagonal calls retain the derivative formula. An empty orthogonal basis gives the zero kernel. See the [API reference](api.md#orthogonal-polynomial-kernel-evaluation) for conditioning and timing scope.
    - **Gap Probabilities & Observables**: Evaluate exact discrete gap probabilities and approximate continuous Fredholm determinants via Nyström discretization, plus exact $k$-point correlation functions.
    - **HKPV Sampler**: Sample real symmetric DPP correlation kernels using floating-point spectral decomposition and Gram-Schmidt projections. Nonprojection kernels use Bernoulli eigenvector selection. Matrix shape, finiteness, symmetry and spectrum in $[0,1]$ are validated with $10^{-10}$ absolute roundoff tolerance.


## Installation

This project utilizes `hatchling` and `hatch-cython` to automatically compile Cython extension modules (`modular_fast.pyx`) on install, conforming to PEP 517/621.

### Requirements

- **Python**: `3.9` or higher
- **C Compiler**: A working C compiler (e.g., `gcc` or `clang`) must be available on your system to compile the Cython modules.

### Setup Instructions

1. Clone the repository and initialize a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the package locally:
   * **Standard Install**:
     ```bash
     pip install .
     ```

   * **Development / Testing Install** (includes Cython source compilation in editable mode and dev tools like `pytest`):
     ```bash
     pip install -e ".[dev]"
     ```

- `numpy >= 1.24`
- `sympy >= 1.12`
- `python-flint >= 0.6.0`
- `scipy >= 1.10`
- `cupy` *(Optional, required for GPU-accelerated root finding)*

*Note on Arbitrary-Precision Dependency*: While `python-flint >= 0.6.0` acts as the primary computational engine for strict real and complex root isolation, the user API does not require passing `flint.fmpq_poly` or specialized objects directly. Standard Python lists and NumPy arrays are automatically cast internally to arbitrary-precision environments where necessary.

### Usage Guide

??? "1. Polynomial Representation & Transformations"

    You can construct polynomials exactly from sequences of rational coefficients and evaluate their roots using numerical eigensolvers or Arb isolation.

    ```python
    from finitefree.core import RealRootedPolynomial
    import sympy as sp

    # Initialize exactly via rational/integer coefficients: (x - 1)(x - 2) = x^2 - 3x + 2
    p = RealRootedPolynomial([1, -3, 2])

    # Dilate, shift, and take positive integer root powers exactly
    p_dilated = p.dilation(2)             # x^2 - 6x + 8 (roots scaled by 2)
    p_shifted = p.shift(1)                # (x - 2)(x - 3) = x^2 - 5x + 6 (roots + 1)
    p_powered = p.power(2)                # (x - 1)(x - 4) = x^2 - 5x + 4 (roots squared)
    p_reversed = p.reversed_polynomial()  # x^2 - 1.5x + 0.5 (reciprocal roots)

    # Domain properties use coefficient signs after real-rootedness verification
    print(p.has_non_negative_roots)       # True
    print(p.has_strictly_positive_roots)  # True

    # Perform certified complex interval root isolation (Arb)
    roots = p.evaluate_roots_float64(exact=True)
    print(roots)  # [1.0, 2.0]

    # Use stable numerical roots for asymptotic comparisons of known families
    from finitefree import gue_expected_poly
    gue_roots = gue_expected_poly(100).evaluate_roots_float64(exact=False)
    assert len(gue_roots) == 100
    ```


??? "2. Finite Free Convolutions"

    Convolutions combine polynomial coefficients exactly. Additive convolution preserves real-rootedness; multiplicative convolution does so when one real-rooted input has non-negative roots. Formal coefficient formulas can also produce polynomials with complex roots, so reconstructed outputs are verified lazily.

    ```python
    from finitefree.core import RealRootedPolynomial
    from finitefree.convolutions import symmetric_additive, asymmetric_additive, multiplicative
    import sympy as sp

    # Instantiate two real-rooted polynomials of degree d=2
    p = RealRootedPolynomial([1, -3, 2])  # roots: 1, 2
    q = RealRootedPolynomial([1, 0, -4])  # roots: -2, 2

    # Symmetric Additive Convolution (p [+]_d q)
    res_add = symmetric_additive(p, q, d=2)
    print(res_add.coeffs)  # [1, -3, -2]

    # Asymmetric Additive Convolution (p [u]_d q) with fractional rank weights
    res_asym = asymmetric_additive(p, q, weights=[sp.Rational(1, 2), sp.Rational(1, 2)], d=2)
    print(res_asym.coeffs)

    # Multiplicative Convolution (p [*]_d q)
    res_mult = multiplicative(p, q, d=2)
    print(res_mult.coeffs)  # Hadamard-like projection
    ```


??? "3. Finite Transforms & Free Cumulants"

    Finite free probability transforms compute expected spectral properties and algebraic limits without combinatorial partition search.

    ```python
    from finitefree import FiniteRTransform, FiniteTTransform, SymmetricFiniteSTransform
    from finitefree.orthogonal import laguerre_polynomial

    # Initialize a polynomial (must be non-negative rooted for T-transform)
    poly = laguerre_polynomial(n=3, alpha=1)

    # Compute Finite Free Cumulants exactly via generating function recurrences (O(d^2))
    # Additivity holds: kappa(p [+] q) = kappa(p) + kappa(q)
    r_transform = FiniteRTransform(poly)
    cumulant_3 = r_transform[2]
    print(f"3rd Finite Free Cumulant: {cumulant_3}")

    # Map inverse limit points using the Fujie-Ueda Finite T-Transform
    t_transform = FiniteTTransform(poly)
    print(t_transform(0.5))
    ```

    Finite free cumulants use the normalization in [Definition 2.14 of Arizmendi et al.](https://arxiv.org/html/2408.09337v2#S2.SS5), $\kappa_n^{(d)}=(-d)^{n-1}c_n/(n-1)!$, where $c_n$ is the classical cumulant of the normalized coefficient sequence. `additive_power` uses the matching inverse. Orders above the ambient dimension return zero as an API convention; they are outside the finite cumulant definition.

    `normalized_coeffs(d)` divides out the leading coefficient before applying the binomial normalization, so `e_0=1` even when a polynomial was stored with `monic=False`. Finite cumulants and coefficient convolutions therefore depend on the roots rather than a scalar multiple of the polynomial. The stored coefficients and polynomial evaluations retain that scalar; reconstructed/convolved results are monic. Larger ambient dimensions pad roots with zeros.

    `FiniteRTransform(..., numerical=True, prec=256)` centers the requested coefficient prefix in rational arithmetic, computes the recurrence with Arb, and restores the first cumulant (the mean). This avoids cancellation from large translations without shifting the entire polynomial or computing all cumulants exactly. Results are floats at the requested working precision; high-order cancellation can still require more precision. `scripts/benchmark_transforms.py` compares shifted Hermite/Wishart cumulants with their analytic values and reports transform time separately from construction.

    `FiniteTTransform` requires positive degree and non-negative roots, and evaluates a right-continuous step function for finite real inputs in $(0,1)$. Rational inputs retain their exact position, including values arbitrarily close to an endpoint. Floats select intervals using their stored binary values; use `sympy.Rational(k, d)` when an exact grid boundary is intended.

    Reconstruction from normalized coefficients produces a formal polynomial with lazy real-rootedness validation. Root extraction and positive-root domain checks reject complex-rooted inputs; formal coefficient convolutions remain available.

    Jacobi and Laguerre polynomials constructed outside their orthogonality domains are also validated lazily. Formal construction remains available, but a root request rejects a complex-rooted result such as `laguerre_polynomial(2, -3)`.


??? "4. Orthogonal Families & Ensembles"

    FiniteFree builds classical and symmetric orthogonal systems exactly via optimized recursive relations.

    ```python
    from finitefree.orthogonal import (
        jacobi_polynomial,
        hahn_polynomial,
        jack_polynomial,
        hermite_polynomial,
    )
    from finitefree.ensembles import sample_gue, gue_expected_poly

    # Construct Jacobi, Hahn, and Hermite recurrence relations exactly over Q
    h_prob = hermite_polynomial(n=4, physicist=False)  # Probabilist Hermite He_4
    jacobi = jacobi_polynomial(n=3, alpha=1, beta=1)     # Jacobi P_3^(1, 1)
    hahn   = hahn_polynomial(n=2, alpha=1, beta=1, N=5)  # Hahn Q_2

    # High-performance multivariate Jack polynomials using sparse exponent dicts
    jack = jack_polynomial(m=3, partition=[2, 1], alpha=2)
    print(jack.expr)

    # Compare random matrix characteristic polynomials with theoretical sequences
    M = sample_gue(d=4)
    expected_poly = gue_expected_poly(d=4)
    print(expected_poly.coeffs)
    ```


??? "5. Multivariate Matrix Pencils"

    Evaluate homogeneous determinants $\det(x_1 A_1 + \dots + x_m A_m)$ exactly via modular matrix interpolation.

    Matrix-pencil constructors copy integer/rational entries before preparing a separate `float64` numerical view. Characteristic polynomials, rational SLP derivatives and multivariate determinants use the copied values, including integers above $2^{53}$. Float inputs preserve their supplied binary values; precision lost before construction cannot be recovered. Matrices must be square, share one shape, contain finite real rational values that fit the finite numerical view, and be symmetric on the supplied values for `SymmetricMatrixPencil`. Construct a new pencil to change its entries.

    ```python
    from finitefree.hyperbolic import SymmetricMatrixPencil
    from finitefree.multivariate import MultivariatePolynomial
    import numpy as np

    # Define a matrix pencil
    A1 = np.eye(3)
    A2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    pencil = SymmetricMatrixPencil([A1, A2])

    # Construct the multivariate polynomial exactly using modular determinants and CRT
    poly_crt = MultivariatePolynomial.from_symmetric_matrix_pencil_interpolated(pencil)
    print(poly_crt.expr)

    # Or utilize Zippel's randomized sparse interpolation over finite fields
    poly_sparse = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(pencil)
    print(poly_sparse.expr)
    ```


??? "6. Determinantal Point Processes (DPPs)"

    Construct correlation kernels, evaluate k-point joint intensities exactly, compute Fredholm determinant gap probabilities, and sample point configurations numerically.

    ```python
    from finitefree import (
        DiscreteFiniteKernel,
        OrthogonalPolynomialKernel,
        hermite_polynomial,
        gap_probability_discrete,
        gap_probability_continuous,
        sample_discrete,
    )
    import numpy as np
    import math
    import sympy as sp

    # --- 1. Discrete DPP Kernel & Exact Gap Probability ---
    # 3x3 nonprojection correlation kernel with eigenvalues 1, 1, 1/3
    K_mat = [
        [sp.Rational(2, 3), sp.Rational(1, 3), 0],
        [sp.Rational(1, 3), sp.Rational(2, 3), 0],
        [0, 0, 1],
    ]
    discrete_kernel = DiscreteFiniteKernel(K_mat)

    # Exact gap probability on states {0, 1}: det(I - K_{[0,1]})
    prob_discrete = gap_probability_discrete(discrete_kernel, [0, 1])
    print(f"Exact Gap Probability on {{0,1}}: {prob_discrete}")  # 0

    # --- 2. Orthogonal Polynomial Kernel & CD Formula ---
    # Probabilist's Hermite polynomials for GUE / Wigner ensemble DPP
    norms = [1, 1, 2] # h_0, h_1, h_2
    polys = [
        hermite_polynomial(0, physicist=False),
        hermite_polynomial(1, physicist=False),
        hermite_polynomial(2, physicist=False),
        hermite_polynomial(3, physicist=False),
    ]
    hermite_kernel = OrthogonalPolynomialKernel(polys, norms)

    # Evaluate kernel off-diagonal and diagonal (CD derivatives)
    print(f"CD Kernel K(0.5, 1.2): {hermite_kernel(0.5, 1.2)}")
    print(f"CD Kernel Diagonal K(0.5, 0.5): {hermite_kernel(0.5, 0.5)}")

    # --- 3. Continuous Fredholm Determinant ---
    # Hermite orthogonal polynomial weight function
    def weight_func(x):
        return math.exp(-x**2 / 2.0) / math.sqrt(2.0 * math.pi)

    gap_prob = gap_probability_continuous(hermite_kernel, -1.0, 1.0, n_points=20, weight_func=weight_func)
    print(f"Continuous Gap Probability over [-1.0, 1.0]: {gap_prob}")

    # --- 4. HKPV Discrete Sampling ---
    sampled_states = sample_discrete(discrete_kernel, state_space=[0, 1, 2])
    print(f"HKPV Sampled point configuration: {sampled_states}")
    ```


## Testing Protocol

Polynomial construction copies caller inputs, and `coeffs` returns an independent editable array. Normalized coefficient arrays and cached real/unitary roots are read-only; call `.copy()` before editing them. This preserves verification and cached results when caller data changes. See the [ownership contract](api.md#coefficient-ownership-and-cached-arrays).

Linear root verification and extraction use the coefficient ratio, including numeric symbolic roots such as `sqrt(2)`. Nonreal roots are rejected, unknown symbolic realness/signs remain uncertified, and nonnumeric or out-of-range numerical root requests fail explicitly. See [real-rootedness](api.md#real-rootedness).

FiniteFree ships with a consolidated robust verification suite designed to run under `pytest`. 

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib tests/ scripts/tests/
```

Documentation contributors can run `python scripts/sync_docs.py` followed by `mkdocs build --strict`. The synchronizer expects single-level `<details>` blocks with opening, summary and closing tags on separate lines. It preserves fenced examples, including literal HTML and links, while converting prose links from the README's `docs/` prefix. Missing input or malformed details blocks abort before changing the generated index. CI checks documentation tools with Ruff, strict mypy and `scripts/tests/` regressions.

- **`test_core.py`**: Real-rootedness verification, divide-and-conquer root synthesis, and sequence extractions.
- **`test_transformations.py`**: Exact algebraic variable scaling (dilation), shifts, powers, and reciprocal-root polynomial transformations.
- **`test_convolutions.py`**: Additive and multiplicative explicit formulas and hyperbolic geometry preservation.
- **`test_transforms.py`**: Exact recursive generating functions and high-order finite free cumulant strict additivity ($\kappa_n^{(d)}$).
- **`test_empirical.py`**: Expected characteristic polynomial identities for GOE ($\beta=1$), GUE ($\beta=2$), and GSE ($\beta=4$) random matrix ensembles using the `ensembles` module.
- **`test_hyperbolic.py`**: Multivariate homogeneous polynomials, CRT grid interpolations, sparse FLINT arrays, and Jacobi SLP evaluations.
- **`test_orthogonal.py`**: Exact hypergeometric and multivariate orthogonal polynomial families (Jacobi, Hahn, Jack).
- **`test_numerical_roots.py`**: Independent 192-bit Arb comparisons, tiny positive hard-edge roots, affine scales, solver fallback and cache behavior.
- **`test_transform_stability.py`**: Shifted Hermite/Wishart cumulants, tiny spreads beneath large translations, ambient dimensions, exact T-transform interval boundaries and the Marchenko–Pastur diagonal limit.
- **`test_dpp.py`**: Correlation-kernel validation, state-space ordering and seeded projection/nonprojection sampling laws.

For reproducible root benchmarks, run `PYTHONPATH=. python scripts/benchmark_roots.py --output roots-benchmark.json`. The script compares uncached numerical roots with independently isolated 192-bit Arb roots at degrees 32, 100 and 300, and reports construction time separately. On jon-desktop (Python 3.13, NumPy 2.2.4, SciPy 1.15.3, python-flint 0.9.0), degree-300 results were:

| Family | Numerical roots | Arb roots | Maximum scaled error |
| :--- | ---: | ---: | ---: |
| GUE additive convolution | 1.29 ms | 1.24 s | $8.5\times10^{-16}$ |
| Square Wishart | 0.97 ms | 17.45 s | $2.6\times10^{-15}$ |
| Rectangular Wishart ($n=2d$) | 0.99 ms | 21.64 s | $3.6\times10^{-15}$ |
| Legendre | 1.23 ms | 1.68 s | $7.8\times10^{-16}$ |
| Laguerre ($\alpha=-1+10^{-12}$) | 0.95 ms | 17.47 s | $4.6\times10^{-15}$ |
| Jacobi ($\alpha=-1+10^{-12}$, $\beta=3/2$) | 1.27 ms | 6.86 s | $1.4\times10^{-15}$ |

Scaled error means $\max_i|\hat\lambda_i-\lambda_i|/\max(1,\max_i|\lambda_i|)$. The smallest Laguerre root was $3.33\times10^{-15}$, with relative error $3.1\times10^{-13}$. Timings are measurements on this machine, not CI thresholds; they exclude polynomial construction. Construction at degree 300 took 5.8–191 ms across these cases.

## Computational Complexity & Architecture

FiniteFree is architected to bypass the combinatorial bottlenecks inherent in high-order differential operators, combinatorial partition counts, and eager root validation. It achieves this by executing convolutions, algebraic transforms, and matrix interpolations directly on polynomial coefficient sequences in C, leveraging `python-flint`'s arbitrary-precision integer/rational arithmetic.

### Complexity Matrix of Key Operations

| Operation | Mathematical Method | Time Complexity | Arithmetic Space |
| :--- | :--- | :---: | :---: |
| **Polynomial Multiplication** | C-level FFT / Kronecker substitution | $O(d \log d)$ | Exact $\mathbb{Q}$ |
| **Root Reconstruction (`from_roots`)** | Binary divide-and-conquer splitting tree | $O(d \log^2 d)$ | Exact $\mathbb{Q}$ |
| **Symmetric Additive Convolution ($\boxplus_d$)** | EGF coefficient multiplication | $O(d \log d)$ | Exact $\mathbb{Q}$ |
| **Asymmetric Additive Convolution ($\uplus_d$)** | Cauchy product of scaled sequences | $O(d \log d)$ | Exact $\mathbb{Q}$ |
| **Multiplicative Convolution ($\boxtimes_d$)** | Pointwise multiplication of normalized coefficients | $O(d)$ | Exact $\mathbb{Q}$ |
| **Sturm Real-Rootedness Verification** | Subresultant Polynomial Remainders Sequence (PRS) | $O(d^2)$ | Exact $\mathbb{Q}$ |
| **High-Precision Root Isolation (Arb)** | Adaptive complex ball arithmetic | Degree, conditioning and precision dependent | Interval $\mathbb{C}$ |
| **Orthogonal-Family Root Approximation** | Symmetric tridiagonal eigenvalues, $O(d)$ matrix storage | $O(d^2)$ | Float $\mathbb{R}$ |
| **General Root Approximation** | Dense balanced companion matrix eigenvalues | $O(d^3)$ | Float $\mathbb{C}$ |
| **Finite R-Transform (Cumulants)** | Generating function recurrence relation | $O(d^2)$ | Exact $\mathbb{Q}$ |

### Architectural Design Principles

#### 1. Exact-to-Approximate Hybrid Pipeline
All algebraic operations, polynomial recurrences, and convolutions are computed in exact rational arithmetic ($\mathbb{Q}$) using GMP/FLINT backends (`fmpq_poly`). Floating-point approximations are deferred entirely to the final egress stage (e.g. root isolation or evaluation), preventing early-stage rounding errors and numerical drift from compounding during intensive convolution chains.

For asymptotic root comparisons, known orthogonal families carry their three-term recurrence alongside the exact polynomial. The numerical path computes parameter differences before float conversion and scales the Jacobi matrix without centering, preserving tiny positive hard-edge roots. Affine shifts are applied after solving. This follows the [Jacobi-matrix characterization of zeros](https://dlmf.nist.gov/18.2#vi) and [classical recurrences](https://dlmf.nist.gov/18.9) using [SciPy's tridiagonal eigenvalue solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvalsh_tridiagonal.html). Arbitrary convolutions, including compound-Wishart lognormal examples, do not inherit unproven recurrence metadata and retain the general solver/reference path. Nonfinite or underflowed recurrences fall back; extreme affine shifts can still lose differences that `float64` cannot represent.

When domain validation uses Arb, it retains the isolated roots for subsequent evaluation. Generic root calls can therefore return the validation result directly. An `exact=True` request at a higher working precision refreshes the root cache; `exact=False` can reuse any available root result. Small-degree generic calls still use exact Sturm validation and may be dominated by coefficient growth. `scripts/benchmark_compound_roots.py` measures first public root calls, including validation, against independent higher-precision Arb roots, with construction and repeat-cache latency reported separately.

For compound-Wishart polynomials with degree $d$, $n=d^2$, and $d$ identical multiplicative factors, measurements on the same jon-desktop environment at 192-bit working precision were:

| Degree | Construction | First root call (`exact=False`, including validation) |
| ---: | ---: | ---: |
| 30 | 0.8 ms | 0.967 s |
| 60 | 6.3 ms | 0.052 s |
| 100 | 60.0 ms | 0.237 s |
| 150 | 304.4 ms | 1.403 s |

Root timings are the best of three fresh calls, excluding construction and library warm-up. Degree-60–150 results matched the independently isolated 384-bit reference after float64 conversion. Degree 10 retained the companion-matrix path with maximum scaled error $5.2\times10^{-13}$; degree 30 was dominated by Sturm verification. Runtime depends on coefficient size and the validation backend as well as degree. No orthogonal-family recurrence is inferred for these compound polynomials.

#### 2. Algebraic Domain Verification (Sturm PRS)
The library verifies real-rootedness lazily. Through degree 30 it uses square-free factorization and exact Sturm sequences, with a subresultant Polynomial Remainder Sequence (PRS) for factors of degree at least 15. Higher degrees use Arb isolation. An imaginary ball merely containing zero is not a certificate; complex roots raise `ValueError`, and inability to certify raises `RuntimeError`. An explicit `assume_real_rooted=True` trusts the caller and bypasses verification.

#### 3. Partition-Free Cumulant Recurrences
Rather than explicitly constructing combinatorial structures (such as enumerating non-crossing partitions to calculate free cumulants), FiniteFree solves the finite $R$-transform and $S$-transform relationships using direct generating function recurrences. By rewriting the underlying algebraic equations into coefficient-level recurrence relations, the combinatorial explosion is reduced to a deterministic $O(d^2)$ exact rational arithmetic sweep.

#### 4. High-Performance Multivariate Matrix Pencil Interpolation
To evaluate multivariate pencils of the form $\det(x_1 A_1 + \dots + x_m A_m)$ and compute exact characteristic polynomials without symbolic expansion blowups, the library avoids symbolic determinant bottlenecks via the following complementary strategies:

* **Reverse-Mode Automatic Differentiation**: Replaces numerical approximations with an exact, linear-time Reverse-Mode AD engine for straight-line programs (SLPs) over $\mathbb{Q}$ to evaluate gradients and Hessians.
* **Exact Rational Interpolation**: Computes characteristic polynomials of matrix pencils via exact rational interpolation over $\mathbb{Q}[t]$.
* **Cython-Accelerated Modular Determinants**: Matrix evaluations are mapped to machine-precision finite fields $\mathbb{F}_p$ for fast C-level Gaussian elimination.
* **Chinese Remainder Theorem (CRT) Reconstruction**: Coefficients computed over multiple distinct prime fields are reconstructed back to exact large integers/rationals over $\mathbb{Q}$.
* **Zippel's Sparse Polynomial Interpolation**: Instead of using an exponential dense grid (which requires $O(n^m)$ points), Zippel's randomized algorithm discovers the non-zero monomial support of the polynomial step-by-step over finite fields, drastically reducing evaluation costs for sparse pencils.

## References

The theoretical architecture and exact computational operators implemented in FiniteFree are grounded in the following foundational literature across finite free probability, classical asymptotic free probability, and random matrix theory.

### Finite Free Probability

* Arizmendi, O., Fujie, K., Perales, D., & Ueda, Y. (2026). *S-transform in finite free probability*. *Advances in Mathematics*, 489, 110803.
* Marcus, A., Spielman, D. A., & Srivastava, N. (2015). *Interlacing families I: Bipartite Ramanujan graphs of all degrees*. *Annals of Mathematics*, 182(1), 307–325.
* Marcus, A., Spielman, D. A., & Srivastava, N. (2022). *Finite free convolutions of polynomials*. *Probability Theory and Related Fields*, 182(3–4), 807–848.

### Classical Free Probability & Asymptotic Limits

* Nica, A., & Speicher, R. (2006). *Lectures on the Combinatorics of Free Probability*. Cambridge University Press.
* Tucci, G. H. (2010). *Limit laws for geometric means of free random variables*. *Indiana University Mathematics Journal*, 59(1), 1–13.
* Voiculescu, D. V., Dykema, K. J., & Nica, A. (1992). *Free Random Variables*. American Mathematical Society.

### Orthogonal Polynomials & Random Matrix Ensembles

* Anderson, G. W., Guionnet, A., & Zeitouni, O. (2010). *An Introduction to Random Matrices*. Cambridge University Press.
* Dumitriu, I., & Edelman, A. (2002). *Matrix models for beta ensembles*. *Journal of Mathematical Physics*, 43(11), 5830–5847.
* Forrester, P. J. (2010). *Log-Gases and Random Matrices* (LMS-34). Princeton University Press.
* Macdonald, I. G. (1995). *Symmetric Functions and Hall Polynomials* (2nd ed.). Oxford University Press.
* Mehta, M. L. (2004). *Random Matrices* (3rd ed.). Elsevier.
* Szegő, G. (1975). *Orthogonal Polynomials* (4th ed., Vol. 23). American Mathematical Society, Colloquium Publications.
