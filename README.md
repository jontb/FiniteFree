# FiniteFree: Finite Free Probability Library

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

FiniteFree is a precision-focused Python framework designed to operationalize the discrete calculus of finite free probability. It extends classical free probability limits to finite-dimensional polynomial representations (characteristic polynomials), mapping linear deterministic operators against polynomial distributions.

To prevent numerical floating-point drift and avoid the computational complexity of high-order runtime differential operators, FiniteFree implements exact finite free convolutions using discrete algebraic representations, exact generating function recurrences, and arbitrary-precision integer and rational scaling.

## Features

- **Exact Real-Rooted Polynomial Validation**: Verified lazily via exact rational Sturm sequences.
- **Unitary Circle Geometries ($\mathbb{T}$)**: Implements `UnitaryPolynomial` structures for polynomials with roots strictly on the complex unit circle, bypassing real-line Sturm sequence constraints and isolating angular arguments via complex eigensolvers (e.g., `unitary_hermite_polynomial`).
- **Lazy Geometric Domain Properties**: $O(d)$ lazy algebraic root verification properties (`has_non_negative_roots` and `has_strictly_positive_roots`) evaluated directly on the coefficients using Descartes' Rule of Signs to enforce operators domains without root seeking.
- **Basic Polynomial Transformations**: Supports exact algebraic transformations including variable dilation (`dilation`), variable shift (`shift`), root powers (`power`), root-reciprocal reversing (`reversed_polynomial`), derivative (`derivative`), projection (`projection`), fractional additive convolution power (`additive_power`), and the Fujie-Ueda limiting polynomial $\Phi_d$ (`phi_d`).
- **Orthogonal Polynomial Families**:
  - **Jacobi Polynomials** (`jacobi_polynomial`): Exact $O(n^2)$ recurrence construction of $P_n^{(\alpha, \beta)}(x)$ using `flint.fmpq_poly` in exact rational arithmetic.
  - **Hahn Polynomials** (`hahn_polynomial`): Exact $O(n)$ recurrence construction of $Q_n(x; \alpha, \beta, N)$ using sequential running products to prevent redundant Pochhammer factorial operations.
  - **Jack Polynomials** (`jack_polynomial`): High-performance variables recursion computing symmetric Jack polynomials $J_\lambda^{(\alpha)}(x_1, \dots, x_m)$ strictly over sparse monomial exponent dictionaries, completely bypassing SymPy's slow symbolic expansion engine.
  - **Chebyshev & Legendre Polynomials** (`chebyshev_t_polynomial`, `chebyshev_u_polynomial`, `legendre_polynomial`): Exact recurrence constructions of $T_n(x)$, $U_n(x)$, and $P_n(x)$ over $\mathbb{Q}$ via `flint.fmpq_poly`.
- **High-Degree Scaling & Root Reconstruction**:
  - **Divide-and-Conquer Polynomial Synthesis**: `RealRootedPolynomial.from_roots(roots)` executes a binary splitting tree algorithm operating in $O(d \log^2 d)$ time for high-speed, exact polynomial synthesis.
- **Finite Free Convolutions**:
  - **Symmetric Additive ($\boxplus_d$)**: Exact analytical evaluation mapping polynomial convolutions against the symmetric finite combinatorial variance.
  - **Asymmetric Additive ($\uplus_d$)**: Validated combinatorial operator supporting mixed rank geometry via fractional squared weights.
  - **Multiplicative ($\boxtimes_d$)**: Exact discrete multiplicative root transformations via scaled Hadamard projections.
- **Analytical Finite Transforms**:
  - **Finite Cauchy Transform** ($G_p^{(d)}$)
  - **Finite S-Transform** ($S_p^{(d)}$): Discrete normalized evaluations bypassing non-linear mapping.
  - **Finite R-Transform** ($R_p^{(d)}$): Computes finite free cumulants ($\kappa_n^{(d)}$) via an exact $O(n^2)$ recursive generating function sequence map over $\mathbb{Q}$, bypassing exponential partition lattice enumeration while verifying exact additivity $\kappa_n^{(d)}(p \boxplus_d q) = \kappa_n^{(d)}(p) + \kappa_n^{(d)}(q)$.
  - **Finite T-Transform** ($T_p^{(d)}$): Step function mapping the right-continuous inverse to the Fujie-Ueda limit $\Phi_d$, evaluated in $O(d)$ algebraically using coefficient sign-alternation validation.
  - **Symmetric Finite S-Transform** ($\tilde{S}_p^{(2d)}$): Evaluates the discrete S-Transform over symmetric domains via exact root-squaring maps ($\mathbf{Sq}(p)$), bypassing zero-valued odd coefficients.

- **Multivariate Hyperbolic Geometry & Matrix Pencils**:
  - **`MultivariatePolynomial`**: Homogeneous multivariate polynomials with exact directional derivatives, mixed partials, and homogeneous multinomial normalization.
  - **Compiled Sparse Evaluations**: Features `to_fmpq_mpoly()` for $O(1)$ evaluation and substitution using compiled C-level sparse representations inside the FLINT library.
  - **Jacobi SLP Operations**: Straightline programs evaluating determinant gradients and Hessians via Jacobi's determinant derivative formulas.
  - **Product-Grid Modular Interpolation (CRT)**: Evaluates exact determinant polynomials for symmetric matrix pencils ($n \ge 4$) modulo prime sequences using C-level `nmod_mat` solvers, reconstructing the integer coefficients via the Chinese Remainder Theorem for a 24x speedup over symbolic equivalents.
  - **Zippel Sparse Interpolation**: Deploys Zippel's probabilistic algorithm over $\mathbb{F}_p$ for sparse determinant evaluations (`from_symmetric_matrix_pencil_sparse`), bounding interpolation complexity to the target monomial count rather than the maximum total-degree combinatorial grid.
  - **Multiplicative Pencils & Diagonal Specialization**: Extends exact matrix pencil geometries to generalized asymmetric forms (`MultiplicativeMatrixPencil`) and computes univariate characteristic projections via optimized 1D Chinese Remainder Theorem loops.
  - **LMI Cone Verification**: Positive definiteness checks ($A(e) \succ 0$) to verify hyperbolic cones.
- **Optimized Memory Pool Management**: Prevents FLINT/GMP memory fragmentation at extreme degrees ($d \ge 1000$) through a conditional, threshold-based garbage collection registry inside `PrecisionContext`.
- **Wilkinson-Proof Numerical Egress**:
  - `to_numpy_poly1d()`: Safe rational coefficient float64 castings with warning alerts.
  - `evaluate_roots_float64()`: Uses python-flint's certified complex interval backend (Arb) to isolate roots, ensuring numerical robustness and preventing Wilkinson's phenomenon.
- **Monte Carlo Empirical Validations**: Haar random matrix generators for Unitary $U(d)$ ($\beta=2$), Orthogonal $O(d)$ ($\beta=1$), and block-quaternionic Symplectic $USp(2d)$ ($\beta=4$) ensembles, verifying exact characteristic polynomial expected identities.

## Installation

This project utilizes `hatchling` as the core build backend conforming to PEP 517/621.

1. Clone the repository and initialize a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the package locally:
   ```bash
   pip install .
   ```

### Dependencies
- `numpy >= 1.24`
- `sympy >= 1.12`
- `python-flint >= 0.6.0`
- `scipy >= 1.10`
- `cupy` *(Optional, required for GPU-accelerated root finding)*

*Note on Arbitrary-Precision Dependency*: While `python-flint >= 0.6.0` acts as the primary computational engine for strict real and complex root isolation, the user API does not require passing `flint.fmpq_poly` or specialized objects directly. Standard Python lists and NumPy arrays are automatically cast internally to arbitrary-precision environments where necessary.

## Quickstart

```python
from finitefree.core import RealRootedPolynomial
from finitefree.convolutions import symmetric_additive

# Initialize polynomials natively from standard lists
p = RealRootedPolynomial([1, -3, 2]) # (x-1)(x-2) = x^2 - 3x + 2
q = RealRootedPolynomial([1, 0, -4]) # x^2 - 4

# Execute exact finite free convolution
result = symmetric_additive(p, q, d=2)
print(result.coeffs)
# Output: [1, 0, -5]

# Perform basic transformations
p_dilated = p.dilation(2)          # [Dil_2 p](x) = x^2 - 6x + 8 (roots scaled by 2)
p_shifted = p.shift(1)             # [Shi_1 p](x) = (x-2)(x-3) = x^2 - 5x + 6 (shifted by 1)
p_powered = p.power(2)             # p^(2) = (x-1)(x-4) = x^2 - 5x + 4 (roots squared)
p_reversed = p.reversed_polynomial() # p^(-1) = x^2 - 1.5x + 0.5 (reciprocal roots)

# Construct orthogonal polynomials exactly
from finitefree import (
    jacobi_polynomial,
    hahn_polynomial,
    jack_polynomial,
    hermite_polynomial,
    laguerre_polynomial,
    krawtchouk_polynomial,
    unitary_hermite_polynomial,
    chebyshev_t_polynomial,
    chebyshev_u_polynomial,
    legendre_polynomial,
    FiniteTTransform,
    SymmetricFiniteSTransform,
)
jacobi_poly = jacobi_polynomial(n=2, alpha=0, beta=0)   # Legendre polynomial P_2(x)
hahn_poly = hahn_polynomial(n=1, alpha=1, beta=1, N=4)  # Hahn polynomial Q_1(x)
jack_poly = jack_polynomial(m=2, partition=[2], alpha=3) # Jack polynomial J_[2]^(3)
hermite_poly = hermite_polynomial(n=2, physicist=False) # Probabilist Hermite He_2(x)
laguerre_poly = laguerre_polynomial(n=2, alpha=1)       # Generalized Laguerre L_2^(1)(x)
krawtchouk_poly = krawtchouk_polynomial(n=2, p=0.5, N=4) # Krawtchouk polynomial K_2(x)
cheb_t = chebyshev_t_polynomial(n=2)                    # Chebyshev T_2(x)
cheb_u = chebyshev_u_polynomial(n=2)                    # Chebyshev U_2(x)
legendre_poly = legendre_polynomial(n=3)                # Legendre P_3(x)

# Evaluate Finite T-Transform (valid as laguerre_poly has strictly non-negative roots)
if laguerre_poly.has_non_negative_roots:
    t_transform = FiniteTTransform(laguerre_poly)
    val = t_transform(0.5)

```

## Visual Spectral Convergence Showcase

FiniteFree's exact algebraic convolutions and root isolating engines accurately recover classical limiting distributions as $d \to \infty$. Below are the visualization assets generated natively by the library:

| Wigner Semicircle Convergence ($He_{d} \boxplus_{d} He_{d}$ as $d \to 300$) | Marchenko-Pastur Convergence (Laguerre as $d \to 300$) |
| :---: | :---: |
| ![Wigner Semicircle Convergence](visuals/wigner_semicircle_convergence.gif) | ![Marchenko-Pastur Convergence](visuals/marchenko_pastur_convergence.gif) |

| Interlacing Preservation Under Convolutions |
| :---: |
| ![Interlacing Preservation](visuals/root_interlacing.png) |

## Computational Complexity & Architecture

FiniteFree is architected to bypass the combinatorial bottlenecks inherent in high-order differential operators and eager root validation. 

### Execution Scaling Insights
The architectural execution of FiniteFree strictly conforms to theoretically optimal limits:
- **Lazy Initialization**: Postponing Sturm sequence checks guarantees that object instantiation remains an $O(1)$ operation ($\approx 15 \mu s$).
- **Quadratic Convolution**: Operations such as symmetric additive convolution ($\boxplus_d$) scale at an $O(d^2)$ complexity, executing operations at degree $d=500$ in fractions of a second.
- **Deferred Verification**: Eager Sturm sequence evaluations scale exponentially at $O(d^3)$, visibly diverging on the benchmark trajectories. The lazy architecture successfully isolates this mathematical penalty from the critical operational path.

### Benchmark Hardware Profile
Performance baselines defining the above complexity limits were generated in the following environment:
- **CPU Architecture**: AMD Ryzen 7 1700 Eight-Core Processor (3.0 GHz)
- **System Memory**: 16 GB RAM (DDR4)
- **Execution Environment**: Python 3.13.5 on Linux (64-bit)

## Testing Protocol

FiniteFree ships with a consolidated robust verification suite designed to run under `pytest`. 

```bash
pytest tests/
```

- **`test_core.py`**: Real-rootedness verification, divide-and-conquer root synthesis, and sequence extractions.
- **`test_transformations.py`**: Exact algebraic variable scaling (dilation), shifts, powers, and reciprocal-root polynomial transformations.
- **`test_convolutions.py`**: Additive and multiplicative explicit formulas and hyperbolic geometry preservation.
- **`test_transforms.py`**: Exact recursive generating functions and high-order finite free cumulant strict additivity ($\kappa_n^{(d)}$).
- **`test_empirical.py`**: Expected characteristic polynomial identities for GOE ($\beta=1$), GUE ($\beta=2$), and GSE ($\beta=4$) random matrix ensembles.
- **`test_hyperbolic.py`**: Multivariate homogeneous polynomials, CRT grid interpolations, sparse FLINT arrays, and Jacobi SLP evaluations.
- **`test_orthogonal.py`**: Exact hypergeometric and multivariate orthogonal polynomial families (Jacobi, Hahn, Jack).
