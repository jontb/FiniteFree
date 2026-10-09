# Rational multivariate workflows

This guide describes FiniteFree 0.2.0, available as `finitefree==0.2.0` on [PyPI](https://pypi.org/project/finitefree/0.2.0/). A `MultivariatePolynomial` represents a sparse polynomial over the rationals in an explicit ordered sequence of SymPy symbols. Construction and algebra do not certify stability, hyperbolicity or real-rootedness.

## Construction and exact evaluation

Native FLINT polynomials must have the same variable names in the same order. Inputs are copied; `variables`, `coefficients()` and `to_fmpq_mpoly()` return caller-owned values. Variable names must be distinct, including when SymPy symbols have different assumptions.

```python
import flint
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(x**2 + sp.Rational(1, 3)*y + 2, [x, y])
assert p.evaluate([sp.Rational(1, 2), 3]) == flint.fmpq(13, 4)
q = MultivariatePolynomial.from_coefficients(p.coefficients(), p.variables)
assert q.expr == p.expr
```

`evaluate` accepts integer/rational coordinates and finite floating coordinates interpreted by their stored binary ratios. It returns an exact FLINT rational. Compare with a FLINT rational, or convert explicitly with `sp.Rational(int(value.p), int(value.q))` when comparing to SymPy; SymPy 1.12's implicit FLINT conversion can lose exactness. Coordinate count must match the stored variables. The python-flint 0.9.0 public evaluation path uses its positional callable interface.

## Rational algebra and derivatives

Addition, subtraction, multiplication, unary negation and nonnegative integer powers are supported. Rational scalar operands work on either side. Polynomial operands must have exactly the same ordered symbols; use `reorder_variables` explicitly when their orders differ. Negative/fractional polynomial powers and polynomial division are outside this API.

```python
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(x + y, [x, y])
q = MultivariatePolynomial(x - y, [x, y])
assert (p*q).expr == x**2 - y**2
assert (p**2 - 2*p + 1).expr == sp.expand((x+y-1)**2)
gradient = (p*q).gradient()
hessian = (p*q).hessian()
assert [g.evaluate([0, 0]) for g in gradient] == [0, 0]
assert [[h.evaluate([0, 0]) for h in row] for row in hessian] == [[2, 0], [0, -2]]
```

`partial_derivative(symbol, order=1)`, `mixed_partial_derivative(orders)` and `directional_derivative(direction)` retain exact coefficients. Derivative orders are nonnegative integer indices, including NumPy integer scalars, with booleans rejected. An order above the polynomial's total degree gives zero without an enormous loop. Exact polynomial derivatives can be evaluated at a singular determinant point; determinant/inverse SLP formulas have different singular-point limitations.

## Explicit context reordering and simultaneous substitution

`reorder_variables(variables)` returns an owned polynomial in a permutation of the same SymPy symbols. It permutes sparse exponent coordinates and preserves the expression. It cannot rename, drop or add variables, or change symbol assumptions. Arithmetic continues to reject implicit reordering.

`substitute(mapping, *, variables=None)` replaces symbols **simultaneously**, using FLINT polynomial composition. Keys must be stored SymPy symbols. The target context defaults to the original ordered variables; it is never inferred or shortened. Supply a nonempty, distinct ordered `variables` sequence when renaming, projecting to fewer variables or embedding into a larger context. Every unreplaced source symbol must occur unchanged in that target context.

Replacements may be rational scalars, rational SymPy polynomial expressions in the target symbols, or `MultivariatePolynomial` objects with exactly the target ordered symbols. Reorder a polynomial replacement explicitly when needed. Python/NumPy finite floating scalars retain their own stored binary ratios, including extended precision. Booleans, strings, arrays, complex/irrational coefficients, unknown symbols, rational functions and nonpolynomial expressions are rejected. Wrap native FLINT polynomial replacements in `MultivariatePolynomial` with the intended target symbols explicitly.

Even full scalar substitution returns an owned **constant polynomial** in the target context, rather than a scalar; use `evaluate` to obtain its exact rational value. Empty mappings return independent copies. Composition can increase degree, coefficient sizes and term count substantially; it has no implicit degree or allocation quota. These operations do not certify stability, hyperbolicity or real-rootedness.

```python
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y, u, v = sp.symbols("x y u v")
p = MultivariatePolynomial(x**2 + 3*x*y + y**2, [x, y])
reordered = p.reorder_variables([y, x])
assert reordered.expr == p.expr
assert reordered.evaluate([2, 1]) == p.evaluate([1, 2])
swapped = p.substitute({x: y, y: x})
assert swapped.expr == sp.expand(p.expr.subs({x: y, y: x}, simultaneous=True))
composed = p.substitute({x: u+v, y: u-v}, variables=[u, v])
assert composed.expr == 5*u**2 - v**2
projected = p.substitute({x: 2}, variables=[y])
assert projected.expr == y**2 + 6*y + 4
constant = p.substitute({x: 2, y: 1})
assert constant.variables == [x, y] and constant.evaluate([0, 0]) == 11
```

## Batched numerical evaluation and derivatives

`evaluate_float64` accepts real arrays with shape `(..., variable_count)`, returns a float for one point and an array with the leading batch shape otherwise. An empty batch is valid. It converts the stored coefficients once and evaluates sparse monomials across the batch.

```python
import numpy as np
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(x**2 + y, [x, y])
points = np.array([[1, 2], [3, 4]], dtype=float)
np.testing.assert_array_equal(p.evaluate_float64(points), [3, 13])
assert p.evaluate_float64([1, 2]) == 3.0
```

`gradient_float64(points)` returns an owned float64 array of shape `(..., m)` and `hessian_float64(points)` returns `(..., m, m)`, where `m` is the stored variable count. For one point their shapes are `(m,)` and `(m, m)`. Empty and higher-dimensional batches retain their leading dimensions. Pack already-broadcast coordinate arrays along the last axis; the APIs accept the same single `points` argument as evaluation. Gradient and Hessian axes follow stored variable order, including after reordering. Hessian entries are mirrored from the upper triangle.

```python
import numpy as np
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

u, v = sp.symbols("u v")
p = MultivariatePolynomial(5*u**2 - v**2, [u, v])
points = np.stack(np.broadcast_arrays(np.array([0, 1, 2])[:, None],
                                     np.array([0, 1])[None, :]), axis=-1)
assert points.shape == (3, 2, 2)
np.testing.assert_array_equal(p.gradient_float64(points),
                              np.stack([10*points[..., 0], -2*points[..., 1]], axis=-1))
np.testing.assert_array_equal(p.hessian_float64([0, 0]), [[10, 0], [0, -2]])
assert p.hessian_float64(np.empty((0, 2))).shape == (0, 2, 2)
```

Derivatives are formed exactly before converting their coefficients to float64, then evaluated as sparse polynomials. There is no division by coordinates or matrix inverse, so zero coordinates and singular determinant points are valid. A huge constant can prevent numerical value evaluation while its gradient and Hessian remain representable. Conversely, differentiation can make coefficients exceed float64 range. Coefficient conversion and nonzero exponent coordinates are cached separately for each requested operation.

Numerical work uses blocks of at most 8192 points with contiguous coordinate columns and reused monomial/component buffers. Derivatives retain at most 64 power arrays per block (at most 4 MiB of cached float64 power data); first powers use coordinate views. The power cache is discarded between blocks and calls. Noncontiguous inputs are gathered a block at a time, preserving logical order without flattening a full input copy. Packing columns can use more temporary storage on small batches. Coefficient/derivative caches, input float64 conversion and finite validation also use memory. This bounds the power cache, not total memory or runtime.

Outputs are fully allocated. In particular, a Hessian needs `8 * batch_size * m**2` bytes for its float64 output alone; a gradient needs `8 * batch_size * m`. When the output itself is too large, call the API on user-selected slices and consume each result before moving to the next slice.

```python
import numpy as np
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(x**3 + x*y, [x, y])
points = np.column_stack((np.linspace(-1, 1, 20001), np.ones(20001)))[::-1]
gradient = p.gradient_float64(points)
np.testing.assert_allclose(gradient[:, 0], 3*points[:, 0]**2 + points[:, 1])
np.testing.assert_allclose(gradient[:, 1], points[:, 0])
assert gradient.flags.owndata and not np.shares_memory(gradient, points)
```

All numerical APIs narrow supplied coordinates to float64, including NumPy extended precision. Nonfinite/complex coordinates and invalid final dimensions raise `ValueError`; unrepresentable requested coefficients or nonfinite intermediate arithmetic/results raise `RuntimeError`. Empty batches still validate the requested coefficients. Finite cancellation, underflow and rounding remain possible, and intermediate overflow can occur even when the mathematical result is finite. This is monomial float64 arithmetic, not certified ball arithmetic or a conditioning-aware solver. Large sparse exponents, term counts and coefficient sizes can still be expensive.

## Reproducible scaling comparison

From a source checkout, run `PYTHONPATH=. python scripts/benchmark_multivariate.py --pin-cpu --output results.json` on Linux, or omit `--pin-cpu` on platforms without CPU affinity. The defaults compare 3, 8 and 16 variables, 96 sparse terms and batches of 1, 256 and 4096 points. For numerical derivatives the baseline evaluates preconstructed exact gradient/Hessian polynomials component by component; construction and first coefficient conversion are excluded from interleaved warm-call medians. Separate setup times are reported. Reordering and simultaneous substitution are compared with explicit SymPy reconstruction workflows, and independent SymPy derivatives/composition validate the outputs.

The record includes all timing samples, runtime versions, source hash and CPU affinity. Peak traced memory covers Python/NumPy allocations and outputs rather than total native memory or process RSS. Performance depends on support, dimensions, degrees and batch size; the script establishes measured cases, not a universal speed guarantee.

For direct comparisons with the merged extension baseline, use the expanded study:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/benchmark_multivariate_scaling.py --baseline-ref 45b8400d3331759047cc9ee19904732073b3fad7 --suite smoke --repeats 3 --output smoke.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/benchmark_multivariate_scaling.py --baseline-ref 45b8400d3331759047cc9ee19904732073b3fad7 --pin-cpu --output scaling.json
```

The baseline commit must exist locally; the tool resolves its full SHA and loads that commit's multivariate implementation alongside the current one. The standard suite includes constant/linear, sparse and dense supports; 3, 8 and 16 variables; degree limits 0–12; 96/384-term sparse cases; 256-bit rational coefficients; empty, single, 256-, 4096- and selected 65536-point batches; and strided/broadcast layouts. Sparse degree limits apply per active variable (up to three active variables per term), while dense limits apply to total degree. It tests reordering, identity/permutation/scalar substitution and expanding compositions, recording exact output term counts and coefficient bit sizes.

Numerical outputs are checked against directly differentiated rational monomials at selected points, with full-batch baseline comparisons. Context outputs are checked against independently expanded SymPy expressions. First-object conversion/differentiation, references, tracing and profiling are reported separately from warm-call medians. All interleaved samples, source/script hashes and runtime/thread/affinity settings are preserved. Linux/macOS also run representative large cases in fresh serial subprocesses and report process peak RSS including import/setup contributions. RSS is a high water rather than an isolated temporary allocation measurement; Python tracing omits some native FLINT allocations. `--large-batch 0` omits the large cases and RSS subprocesses. Omit `--pin-cpu` where affinity is unavailable.

An optimization can trade fewer allocations for extra copying or block-loop work. Constant/small batches may receive little benefit, and bounded blocks can lose to whole-batch evaluation for some dense supports. Increasing batch size still scales output storage, increasing sparse degree can increase power cost, and exact expanding composition can multiply support and coefficient bit sizes. These remaining costs are distinct from removed Python reconstruction and per-monomial allocation overhead. Use the raw measurements for the intended workload; no case establishes a universal speed or accuracy bound.

## Exact restriction to a line

`restrict_line(base_point, direction)` forms `P(base_point + t*direction)` using exact polynomial arithmetic and preserves its leading scalar. It returns the existing univariate class with lazy real-rootedness certification. The identically zero restriction raises `ValueError` because that class cannot represent zero.

```python
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
p = MultivariatePolynomial(3*(x+y)**2, [x, y])
q = p.restrict_line([1, 0], [0, 1])
assert list(q.coeffs) == [3, 6, 3]
assert q.verify_real_rootedness()
value = q.evaluate(sp.Rational(1, 3))
assert sp.Rational(int(value.p), int(value.q)) == sp.Rational(16, 3)
```

A generic rational polynomial need not have real-rooted line restrictions:

```python
import sympy as sp
from finitefree.multivariate import MultivariatePolynomial

x, y = sp.symbols("x y")
q = MultivariatePolynomial(x**2+y**2, [x, y]).restrict_line([1, 0], [0, 1])
assert list(q.coeffs) == [1, 0, 1]
try:
    q.verify_real_rootedness()
except ValueError:
    pass
else:
    raise AssertionError("This line restriction has complex roots")
```

## Homogeneous normalization and matrix pencils

`normalized_coefficients()` computes `c_alpha / multinomial(d, alpha)` for a homogeneous degree-`d` polynomial. Nonhomogeneous inputs raise `ValueError`; the zero polynomial returns an empty coefficient mapping. The normalization is an algebraic convention and does not imply a multivariate finite convolution or geometric certificate.

Determinant factories continue to accept the exact rational matrix-pencil snapshots described in [the API](api.md#matrix-pencil-input-contract). Direct, grid-interpolated and sparse constructions can all use the same evaluation/algebra workflows above. Interpolation, coefficient count, degree and bit size can still dominate construction.

After clearing denominators, modular reconstruction continues until the product of accepted primes exceeds twice a determinant coefficient bound. For integer matrices, one such bound is `n! * product_r max_c sum_j abs(A_j[r,c])`: each determinant permutation is a product of linear forms, and the coefficient sum of absolute values is bounded by the product of their coefficient sums. This guarantees a unique centered integer reconstruction of every recovered coefficient. Mere agreement between consecutive CRT reconstructions is insufficient.

The dense grid covers the full homogeneous coefficient support. `from_symmetric_matrix_pencil_sparse` keeps randomized Zippel discovery, then independently verifies all coefficients and the complete support with one exact integer determinant. The coefficient bound alone does not prove support completeness.

For a nonzero bound `B`, dehomogenize the integer determinant by setting the last variable to 1. Encode the other exponents in radix `n+1`, using weights `w_j=(n+1)**j`. Every homogeneous monomial then has a distinct encoded exponent. Evaluate at integer coordinates `x_j=b**w_j`, where `b=2*B+1`. Both the true coefficients and an accepted candidate's coefficients lie in `[-B, B]`. A nonzero difference has coefficients of magnitude at most `b-1`; its highest term at `b` strictly exceeds the sum of every lower term. Equality of the candidate value and the exact determinant therefore proves equality of every coefficient, including monomials omitted by discovery. This is a deterministic identity check, not a random point test.

If the candidate fails its coefficient bounds or this identity check, balanced-base digits of the exact determinant recover the complete polynomial directly. Eight failed prime fields also trigger this exact fallback, avoiding unlimited retries of singular randomized systems. Negative digits, zero coefficients and denominator restoration are handled exactly. A zero row bound proves the determinant identically zero without discovery; an empty matrix has determinant 1.

The encoding can grow exponentially with the variable count. The keyword-only `max_verification_bits` defaults to **1,000,000** and caps the conservative bound `(E+1)*b.bit_length()`, where `E=n*(n+1)**(m-2)` for `m>=2`. This bounds the encoded determinant and evaluated matrix-entry sizes. The check runs before randomized discovery or allocation of the large powers. Exceeding it raises `ValueError`; the method never returns a candidate without verification. Increase the limit explicitly only when the larger integers are feasible, or choose a different determinant constructor. This is an integer-size guard, not a total memory or runtime quota: determinant arithmetic, discovery and fallback decoding can still be expensive. One-variable, empty-matrix and proved-zero-row cases bypass the encoding.

```python
import sympy as sp
from finitefree.hyperbolic import SymmetricMatrixPencil
from finitefree.multivariate import MultivariatePolynomial

pencil = SymmetricMatrixPencil([
    [[1, 0], [0, 0]],
    [[0, 0], [0, 1]],
    [[0, 0], [0, -2]],
])
p = MultivariatePolynomial.from_symmetric_matrix_pencil_sparse(
    pencil, max_verification_bits=1_000_000
)
x, y, z = p.variables
assert sp.expand(p.expr - x * (y - 2*z)) == 0
```

## Prepared numerical pencils (develop)

For repeated numerical queries, call `prepare_numeric()` on a symmetric or
multiplicative pencil. It takes an owned snapshot of the existing float64 matrix
view. No determinant polynomial is expanded, and exact FLINT methods and existing
SLP/polynomial derivative behavior are unchanged.

```python
import numpy as np
from finitefree.hyperbolic import SymmetricMatrixPencil

pencil = SymmetricMatrixPencil([np.eye(2), np.array([[0., 1.], [1., 0.]])])
prepared = pencil.prepare_numeric(point_block_size=64, coefficient_block_size=16)
points = np.array([[2., 0.], [2., 1.]])
factors = prepared.factor(points)
sign, logabsdet = factors.slogdet()
gradient = factors.logabsdet_gradient()
hvp = factors.logabsdet_hvp(np.array([1., 0.]))
assert np.allclose(logabsdet, np.log([4., 3.]))
assert np.allclose(gradient, [[1., 0.], [4./3., -2./3.]])
```

`PreparedMatrixPencil(matrices, backend="numpy")` also accepts an already numeric
resident float64 array `(m,n,n)`, with positive m and n. The object owns both a
coefficient snapshot and packed right-hand sides: approximately `2*m*n*n*8` bytes.
There is deliberately no implicit dtype conversion in this constructor or its
queries. To use CuPy, install the CuPy distribution appropriate for the cloud
GPU's CUDA runtime, select `backend="cupy"`, and transfer point/direction arrays
explicitly with `cupy.asarray`. The pencil's `prepare_numeric(backend="cupy")`
convenience method explicitly transfers the coefficients once. CuPy is optional;
there is no CPU fallback when the GPU backend is requested.

- Coordinates: resident float64 arrays `(...,m)`, including zero coordinates,
  arbitrary leading batch axes, strided inputs and empty batches. Lists, complex,
  object, integer and float32 query arrays are rejected. Convert deliberately
  before repeated calls. Strided reshaping can copy; contiguous inputs avoid it.
- Results: `evaluate` returns `(...,n,n)`; `slogdet` returns two arrays of shape
  `(...)`, including zero-dimensional arrays for a single point. Gradient/HVP
  return `(...,m)`. HVP directions are `(m,)` or exactly the point shape.
- Ownership: prepared coefficients and LU factors do not alias caller inputs.
  Output arrays are caller-owned; cached gradients and slogdet survive edits to
  prior outputs. Retaining a factorization also retains its prepared pencil.
- Numerical domain: finite float64 only. Finite checks occur at API boundaries
  and after assembly/solves to catch overflow. Exact singular LU pivots produce
  `(0,-inf)`; derivatives reject an entire batch containing a singular matrix.
  No pseudoinverse, determinant-magnitude threshold, lower precision or
  regularization is substituted. Near-singular derivatives can be inaccurate
  because of conditioning, and nonfinite results raise `FloatingPointError`.
  These are derivatives of `log(abs(det(A(x))))`, not of the determinant.
- Cone domain: general real nonsingular pencils, including indefinite and
  asymmetric matrices, are supported. Positive determinant is not a test for
  positive definiteness; this milestone supplies no cone-specific operation.
- Reuse: one pivoted LU per point supplies slogdet and all subsequent solves.
  The first HVP caches the gradient using those same coefficient solves.
  Repeated gradient/slogdet queries copy cached results. Repeated HVPs reuse LU
  and packed coefficients but recompute coefficient solves to bound memory.
  No inverse or full Hessian is formed. LU currently dispatches one matrix at a
  time through SciPy or CuPy's public two-dimensional LU interface.
- Blocking: `point_block_size` bounds matrix-assembly scratch, and
  `coefficient_block_size` bounds RHS-solve scratch. Factors retain
  `O(batch*n*n)` storage, gradients `O(batch*m)`, and HVP scratch includes
  `O(coefficient_block_size*n*n)`. Output storage, persistent coefficients,
  backend workspaces and any input-contiguity copy are not covered by block
  sizes. Split very large inputs into external batches to bound total memory.
- Device: CuPy coefficients, factors, inputs and outputs remain on the same
  preparation device. A changed current device or cross-device input is an
  error. Operations use the current stream; callers must order multiple streams.
  Validation and solver status checks can synchronize. No array is implicitly
  copied back to the CPU, but scalar validation decisions are host-visible.

The backend uses the official CuPy
[LU factorization](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.linalg.lu_factor.html)
and [solve](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.linalg.lu_solve.html)
contracts. GPU execution remains unverified in this milestone's CPU-only cloud
environment; see the [validation checklist](development.md#prepared-pencil-validation-checklist).

### Prepared pencil benchmark

Run each backend in a separate process; start with small cases:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/benchmark_prepared_pencils.py --backend numpy --size 16 --variables 4 --batches 1 16 128 --output /tmp/pencils-cpu.json
# Run only on an available cloud GPU with a compatible CuPy installation:
PYTHONPATH=. python scripts/benchmark_prepared_pencils.py --backend cupy --size 16 --variables 4 --batches 1 16 128 --output /tmp/pencils-gpu.json
```

The JSON separates context startup, preparation, coefficient/input/output
transfers, first factor/derivative calls, synchronized warm factorization,
reused-factor HVPs, cached gradients, CUDA-event time, numerical error and memory.
It records failures and never emulates a GPU. Python traced memory and an isolated
CuPy pool high-water allocation are useful partial measurements, **not total
native/device peak memory**; collect external telemetry for that. Compare matching
matrix sizes, batches, dtype, thread limits and accuracy before locating a measured
CPU/GPU crossover. No GPU speedup is established by CPU tests or CPU timings.
