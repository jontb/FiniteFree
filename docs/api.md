# API Reference

This page describes the mathematical and numerical contracts with executable examples, followed by reference sections generated from docstrings and source signatures.

## Root evaluation and conditioning

`RealRootedPolynomial.evaluate_roots_float64(parallel=False, exact=True)` returns sorted NumPy `float64` roots, with multiplicities. It validates real-rootedness before extraction unless the constructor or a proven family has already marked the polynomial verified.

- `exact=False` first uses a symmetric tridiagonal Jacobi matrix when recurrence metadata is available. It scales the matrix without centering, computes parameter differences before float conversion, and accumulates affine parameters rationally before converting the final roots.
- `exact=True` requests python-flint/Arb isolation and remains the default. Use `PrecisionContext(degree=d, prec=192)` to request a working precision. This option selects the high-precision path; the returned floats are neither exact algebraic roots nor interval error bounds. SymPy/general numerical fallbacks may still be used if isolation fails.
- An `exact=True` request bypasses a cache produced with `exact=False`, or a high-precision cache recorded at a lower working precision. An `exact=False` request can reuse any existing root result. Repeated calls may return cached arrays; do not modify those arrays in place.
- When real-rootedness validation uses Arb, it retains the isolated roots and working precision in that same cache. Generic root calls can return this result directly, without another isolation or a companion-matrix pass. Without a suitable cache, dispatch follows the solver options below.
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

For generic compound-Wishart convolutions, `PYTHONPATH=. python scripts/benchmark_compound_roots.py --output compound-roots.json` measures first public root calls with domain validation included, at degrees 10, 30, 60, 100 and 150. It compares 192-bit working precision with an independent 384-bit reference and counts isolation calls. Construction, numerical-library warm-up and repeat-cache latency are excluded from first-call timings and reported separately where applicable. These polynomials retain the general solver rather than inheriting an orthogonal-family recurrence.

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

`numerical=True` uses Arb at `prec` bits (default 256) after centering the requested normalized-coefficient prefix exactly. The first cumulant is restored from the original mean; higher cumulants are invariant under translation. This avoids cancellation from large offsets while keeping the cumulant recurrence numerical. Centering uses the specified ambient dimension, including any zero padding. The precision setting is restored on return or failure. Results are float approximations rather than error bounds; increasing the order or using badly conditioned coefficients may require increasing `prec`.

```python
import numpy as np
from finitefree import FiniteRTransform, gue_expected_poly

p = gue_expected_poly(8).shift(10**50)
values = FiniteRTransform(p, order=4, numerical=True, prec=128)
assert values[0] == 1e50
np.testing.assert_allclose(values[1:], [1, 0, 0], rtol=0, atol=1e-28)
```

For repeatable accuracy and timing checks, run `PYTHONPATH=. python scripts/benchmark_transforms.py --output transform-benchmark.json`. The default cases use degrees 60, 150 and 300, order 12, a shift of $10^{50}$ and 128-bit working precision. Hermite and Wishart analytic cumulants supply the reference. Transform times are best of three with normalized coefficients already cached; construction is reported separately. These cases test stability under translation, not arbitrary high-order inputs.

### Finite T-transform

`FiniteTTransform(p)` requires positive degree and non-negative roots. Its input must be a finite real value in $(0,1)$. It uses the right-continuous intervals in [Definition 6.3 and Remark 6.4](https://arxiv.org/html/2408.09337v2#S6.SS1), including the boundary of an atom at zero. Rational inputs and stored binary float values select intervals exactly, without a float multiplication that could round onto a neighboring grid boundary. Real symbolic constants are also supported. A float approximation to $k/d$ can lie on either side of that rational number; supply the rational value to select the exact boundary.

```python
import sympy as sp
from finitefree import FiniteTTransform, RealRootedPolynomial

T = FiniteTTransform(RealRootedPolynomial.from_roots([1, 2, 3, 4, 5]))
assert T(sp.Rational(1, 10**400)) == sp.Rational(300, 137)
assert T(1 - sp.Rational(1, 10**20)) == 3
assert T(0.6) == sp.Rational(45, 17)  # stored float is below 3/5
assert T(sp.Rational(3, 5)) == sp.Rational(17, 6)
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

## Matrix-pencil input contract

`SymmetricMatrixPencil` and `MultiplicativeMatrixPencil` accept nonempty sequences of equally sized square matrices containing real rational values, including Python/NumPy integers, Python floats, `Fraction`, SymPy rationals and Flint rationals. Constructors copy the entries into rational matrices before creating their separate `float64` views. Supplied floats represent their actual binary values, not a recovered decimal intent. Entries must fit a finite float64 numerical view; tiny entries may underflow in that view while their rational values remain preserved.

`characteristic_polynomial`, rational SLP evaluation/gradient/Hessian, diagonal specialization and direct/interpolated/sparse multivariate determinant construction use the preserved rational values. `evaluate` and numerical SLP operations use the float64 views. Construct a new pencil when changing matrix entries rather than mutating its stored views.

Invalid shape, inconsistent dimensions, empty matrix sequences, nonfinite/complex entries or nonsymmetric supplied entries in `SymmetricMatrixPencil` raise `ValueError`. Symmetry uses the rational entries without an approximate tolerance, so float rounding cannot hide a difference between large integers. Coordinate sequences must match the number of matrices in numerical and rational evaluation. Modular paths reduce integer entries modulo each prime before int64 conversion for Cython calculations.

```python
from finitefree.hyperbolic import SymmetricMatrixPencil

large = 2**53 + 1
pencil = SymmetricMatrixPencil([[[large]]])
assert pencil.characteristic_polynomial([1])[0] == -large
assert pencil.evaluate([1])[0, 0] == float(large)
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
