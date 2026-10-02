# API Reference

This page describes the mathematical and numerical contracts with executable examples, followed by reference sections generated from docstrings and source signatures.

## Root evaluation and conditioning

`RealRootedPolynomial.evaluate_roots_float64(parallel=False, exact=True)` returns sorted NumPy `float64` roots, with multiplicities. It validates real-rootedness before extraction unless the constructor or a proven family has already marked the polynomial verified.

- `exact=False` first uses a symmetric tridiagonal Jacobi matrix when recurrence metadata is available. It scales the matrix without centering, computes parameter differences before float conversion, and accumulates affine parameters rationally before converting the final roots.
- `exact=True` first requests python-flint/Arb isolation and remains the default. Use `PrecisionContext(degree=d, prec=192)` to request a working precision. This option selects the high-precision path; the returned floats are neither exact algebraic roots nor interval error bounds. Existing SymPy/general numerical fallbacks may still be used if isolation fails.
- An `exact=True` request bypasses a cache produced with `exact=False`. An `exact=False` request can reuse an existing high-precision result. Repeated calls may return cached arrays; do not modify those arrays in place.
- If no usable recurrence is available, `exact=False` uses the existing balanced companion-matrix path, or the Aberth–Ehrlich path when `parallel=True`, with Arb/general numerical fallback. Generic monomial coefficients can be ill-conditioned. Fast recurrence metadata is preserved through affine transforms and proven Hermite additive convolutions, not through arbitrary coefficient operations.

The recurrence path is available for degree at least two in these domains:

| Factory or operation | Numerical recurrence coverage |
| :--- | :--- |
| `hermite_polynomial` | Physicist and probabilist conventions |
| `laguerre_polynomial(n, alpha)` | `alpha > -1`; the boundary `alpha == -1` uses the general path |
| `jacobi_polynomial(n, alpha, beta)` | `alpha > -1` and `beta > -1` |
| `legendre_polynomial`, `chebyshev_t_polynomial`, `chebyshev_u_polynomial` | All supported degrees |
| `gue_expected_poly(d)` | Normalized Hermite recurrence |
| `wishart_expected_poly(d, n)` | Positive integer dimensions with `n >= d` |
| `dilation`, `shift` | Inherit an available recurrence; negative dilation is allowed |
| `symmetric_additive(p, q, d)` | Full-degree shifted/dilated Hermite inputs with exact family provenance |

Nonfinite or underflowed recurrence entries and eigensolver failures fall back to the general path. Near-zero positive Laguerre roots require scaling without centering. Extreme scales or shifts can still overflow, underflow or erase root separations that `float64` cannot represent; `exact=True` cannot overcome final-output rounding. No uniform error guarantee is claimed for arbitrary polynomials. Compound-Wishart/lognormal convolutions retain the general solver.

```python
import numpy as np
from finitefree import PrecisionContext, gue_expected_poly
from finitefree.convolutions import symmetric_additive

d = 32
p = gue_expected_poly(d)
result = symmetric_additive(p, p, d)
numerical = result.evaluate_roots_float64(exact=False)
with PrecisionContext(degree=d, prec=192):
    reference = result.evaluate_roots_float64(exact=True)
np.testing.assert_allclose(numerical, reference, atol=1e-13, rtol=1e-13)
assert len(numerical) == d
```

The [reproducible benchmark and measured results](index.md#testing-protocol) compare six families at degrees 32, 100 and 300 against independently isolated Arb roots. Root timings exclude polynomial construction and cached results; construction is reported separately. Degree-300 numerical extraction took 0.95–1.29 ms on jon-desktop with maximum scaled error `4.51e-15`. These measurements are not performance or accuracy guarantees for other inputs or machines.

## Validation and mathematical contracts

### Real-rootedness

`from_normalized_coeffs` requires a nonempty sequence starting with `e_0 == 1` and reconstructs the formal polynomial without certifying its roots. Root extraction and positive-root domain properties validate it lazily. Coefficient sign alternation is used only after real-rootedness is verified. The zero polynomial is rejected; nonzero constants have no roots. `FiniteCauchyTransform` requires positive degree.

Verification uses square-free factorization and exact Sturm sequences through degree 30, with subresultant PRS for factors of degree at least 15. Higher degrees use Arb. An imaginary ball merely containing zero is not accepted as a real-root certificate. Complex-rooted inputs raise `ValueError`; inability to certify raises `RuntimeError`. Passing `assume_real_rooted=True` explicitly trusts the caller's claim and bypasses this verification. Jacobi/Laguerre construction outside a known orthogonality domain remains available but validates lazily.

```python
from finitefree import RealRootedPolynomial

formal = RealRootedPolynomial.from_normalized_coeffs([1, 0, 1])  # x^2 + 1
try:
    formal.evaluate_roots_float64(exact=False)
except ValueError:
    pass
else:
    raise AssertionError("Complex-rooted reconstruction was accepted")
```

### Finite free cumulants

`FiniteRTransform` uses $\kappa_n^{(d)}=(-d)^{n-1}c_n/(n-1)!$, where $c_n$ is the classical cumulant of the normalized coefficient sequence, following [Definition 2.14](https://arxiv.org/html/2408.09337v2#S2.SS5). `additive_power` uses the matching inverse: its polynomial coefficients agree with repeated symmetric additive convolution for positive integer powers. Requested orders above the ambient dimension return zero as an API convention, outside the finite cumulant definition.

```python
from finitefree import FiniteRTransform, RealRootedPolynomial
from finitefree.convolutions import symmetric_additive

p = RealRootedPolynomial.from_roots([0, 0, 3])
assert FiniteRTransform(p, order=3) == [1, 3, 9]
assert list(p.additive_power(2).coeffs) == list(symmetric_additive(p, p, 3).coeffs)
```

### Empirical coefficient comparison

`EmpiricalComparison(p, samples, generator)` stores eigenvalues and characteristic coefficients from at least two samples. Each sample must be a finite Hermitian matrix of shape `(d, d)`, where `d=p.degree`. A `(2d, 2d)` representation is accepted when its sorted eigenvalues occur in pairs; one eigenvalue from each pair supplies the degree-`d` characteristic polynomial. Hermitian and pairing checks permit absolute roundoff up to `1e-10` times the largest entry magnitude or eigenvalue magnitude, respectively. Matrix eigenvalues, sampled coefficients, and analytical coefficients must fit finite `float64` values.

`verify_coefficients(alpha=0.05, *, rtol=1e-10, atol=0)` compares every coefficient mean to the analytical target with half-width

$$
t_{1-\alpha/(2(d+1)),\,N-1}\frac{s}{\sqrt{N}}
+\mathrm{atol}+\mathrm{rtol}\,|\mathrm{target}|.
$$

The Student-t mean band follows the [NIST definition](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm); [Bonferroni adjustment](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm) allocates `alpha` across the `d+1` coefficients without assuming independence between coefficients. Smaller `alpha` gives wider bands. Samples must be independent; Student-t coverage is exact for normal coefficient observations and is a large-sample approximation for other distributions. Characteristic coefficients of random matrices need not be normal. A return value of `True` reports no detected coefficient mismatch under this diagnostic, not proof of a sampler's distribution.

Zero sample variance still requires agreement within the explicit numerical tolerances. The calculation divides each coefficient column and its target by their maximum magnitude before computing the mean and standard deviation, avoiding variance overflow or underflow across different coefficient scales. It cannot recover precision already lost in eigenvalues or characteristic coefficients. Nonfinite/invalid `alpha`, unrepresentable critical values, and negative/nonfinite tolerances raise `ValueError`.

```python
import numpy as np
from finitefree import EmpiricalComparison, RealRootedPolynomial

p = RealRootedPolynomial.from_roots([1, 2])
matching = EmpiricalComparison(p, 3, lambda: np.diag([1, 2]))
assert matching.verify_coefficients()
mismatch = EmpiricalComparison(p, 3, lambda: np.zeros((2, 2)))
assert not mismatch.verify_coefficients()
```

### Discrete DPP sampling

`sample_discrete` uses floating-point spectral HKPV sampling for real symmetric correlation kernels. Nonprojection kernels first select eigenvectors with independent Bernoulli draws. It requires a finite square matrix matching the state-space size, symmetry, and spectrum in `[0, 1]`. Absolute roundoff up to `1e-10` is allowed; small asymmetry is symmetrized and eigenvalues are clipped only within that tolerance. Invalid kernels raise `ValueError` before sampling; loss of the projection basis raises `RuntimeError`.

An ndarray is indexed in the supplied state-space order and may use arbitrary distinct hashable labels. A `BaseKernel` is evaluated on the actual states, so subsets and permutations select the corresponding principal matrix. `DiscreteFiniteKernel` accepts integer indices, including NumPy integers, and rejects fractional indices. Empty state spaces return an empty sample; an ndarray must then have shape `(0, 0)`.

```python
import numpy as np
from finitefree import DiscreteFiniteKernel, sample_discrete

kernel = DiscreteFiniteKernel(np.diag([0.0, 1.0, 1.0]))
assert sample_discrete(kernel, [2, 0]) == [2]
assert sample_discrete(np.diag([0.0, 1.0]), ["excluded", "included"]) == ["included"]
try:
    sample_discrete(2 * np.eye(2), [0, 1])
except ValueError:
    pass
else:
    raise AssertionError("Invalid correlation spectrum was accepted")
```

## Core Operations

::: finitefree.core

## Convolutions

::: finitefree.convolutions

## Transforms

::: finitefree.transforms

## Orthogonal Polynomials

::: finitefree.orthogonal

## Random Matrix Ensembles

::: finitefree.ensembles

## Determinantal Point Processes (DPP)

::: finitefree.dpp

## Multivariate Geometry & Matrix Pencils

::: finitefree.hyperbolic
::: finitefree.multivariate
