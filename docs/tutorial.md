# Numerical workflows and precision

This tutorial uses the **FiniteFree 0.4** APIs, unchanged by the metadata-only 0.4.1 patch. Install with `python -m pip install finitefree==0.4.1`, or follow the [development guide](development.md) for a source checkout. The [API reference](api.md) records numerical precision, ownership and conditioning limits.

## Repeated numerical work

- Pass an explicit NumPy `Generator` with `rng=` for a sampling stream owned by your experiment. Omitting it preserves the global RNG and existing draw order; reproducibility assumes a fixed NumPy/build environment.
- Use `OrthogonalPolynomialKernel.prepare_numeric()` for repeated basis/kernel batches and reusable finite-interval quadrature. Supply the reference measure density explicitly for weighted measures, including Hermite kernels.
- Use `DPPCountDistribution` for finite-rank count PMFs, CDFs, upper tails and moments. Continuous count laws are quadrature estimates with diagnostics, not certified error bounds.
- Use `StreamingEmpiricalComparison` when coefficient moments and configured interval histograms suffice. Keep `EmpiricalComparison` when you need retained samples or plots. See the [seeded GUE streaming example](gue-workflow.md#streaming-interval-summaries) for a direct count reference on the same draws.

See the executable [kernel, count and streaming examples](api.md#optional-numerical-workflows) and [sampling-stream example](api.md#ensemble-random-streams). These are opt-in workflows; existing numerical and algebraic defaults remain available.

## Polynomial inputs and finite cumulants

Sequence coefficients descend from the leading term. FLINT polynomial coefficients use their native ascending order. Construction is monic by default; `monic=False` retains a common scalar. Normalized coefficients and cumulants depend on the roots, while stored coefficients and ordinary evaluation retain the scalar.

```python
import sympy as sp
from finitefree import FiniteRTransform, RealRootedPolynomial
from finitefree.convolutions import symmetric_additive

p = RealRootedPolynomial([2, -6, 4], monic=False)
assert p.evaluate(0) == 4
assert list(p.normalized_coeffs()) == [1, sp.Rational(3, 2), 2]
assert FiniteRTransform(p, order=2) == [sp.Rational(3, 2), sp.Rational(1, 2)]
assert list(symmetric_additive(p, p, 2).coeffs) == [1, -6, sp.Rational(17, 2)]
assert FiniteRTransform(p, order=3)[2] == 0  # Above ambient dimension.
```

## Root extraction and owned arrays

Use `exact=False` for the recurrence-based numerical path of known orthogonal families. `exact=True` requests high-precision isolation but still returns float64 values. Arrays are read-only, and an exact request bypasses an approximate cache.

```python
import numpy as np
from finitefree import PrecisionContext, gue_expected_poly

p = gue_expected_poly(16)
numerical = p.evaluate_roots_float64(exact=False)
with PrecisionContext(degree=16, prec=192):
    reference = p.evaluate_roots_float64(exact=True)
np.testing.assert_allclose(numerical, reference, atol=1e-13, rtol=1e-13)
assert not reference.flags.writeable
edited = reference.copy()
edited[0] = 100
assert p.evaluate_roots_float64()[0] != 100
```

Exact stored coefficients do not prevent numerical overflow, underflow or loss of close root separations. Real-root certification is distinct from numerical extraction: exact construction may remain valid even when a root cannot fit float64.

## Projection and transforms

Proper projection is a monic derivative of order `degree-j` computed from the leading prefix. Full-degree projection preserves object identity. T-transform intervals are right-continuous; use exact rationals for intended grid boundaries.

```python
import sympy as sp
from finitefree import FiniteTTransform, RealRootedPolynomial

p = RealRootedPolynomial.from_roots([1, 2, 3, 4, 5])
assert p.projection(5) is p
assert p.projection(2).degree == 2
T = FiniteTTransform(p)
assert T(0.6) == sp.Rational(45, 17)  # The stored float is below 3/5.
assert T(sp.Rational(3, 5)) == sp.Rational(17, 6)
```

## Exact and sampled moments

`root_moments` and `matrix_moments` share one normalization, so exact root moments and sampled spectra can be compared directly. Compute the moments of each actual draw, then average them. The matrix-level standard error describes sampling spread across independent draws. The per-draw agreement with direct traces is the numerical check. Neither is a deterministic accuracy bound, and expected-polynomial roots do not replace sampled spectra.

```python
import numpy as np
from finitefree import matrix_moments, sample_gue

rng = np.random.default_rng(862)
draws = [sample_gue(4, rng=rng) for _ in range(16)]
moments = np.array([matrix_moments(m, 4) for m in draws])
traces = np.array(
    [[1.0] + [np.trace(np.linalg.matrix_power(m, k)).real / 4 for k in range(1, 5)]
     for m in draws]
)
np.testing.assert_allclose(moments, traces, rtol=1e-12, atol=1e-13)
mean = moments.mean(axis=0)
standard_error = moments.std(axis=0, ddof=1) / np.sqrt(len(draws))
print(np.round(mean, 3), np.round(standard_error, 3))
```

## Exact matrix inputs

Pencils copy rational entries before making separate float64 views. Exact methods preserve values beyond float64's integer precision. Symmetry is checked on supplied values; exact determinant derivatives require a nonsingular evaluated matrix.

```python
from finitefree.hyperbolic import SymmetricMatrixPencil

large = 2**53 + 1
pencil = SymmetricMatrixPencil([[[large]]])
assert pencil.characteristic_polynomial([1])[0] == -large
assert pencil.evaluate([1])[0, 0] == float(large)
```

## Finite-state DPP samples and counts

A discrete DPP here is specified by a real symmetric *correlation* kernel $K$ with spectrum in $[0,1]$. Row and column $i$ of the matrix belong to the $i$-th entry of the supplied state space, so with labels `[101, -7]` the first row is state `101`. The kernel gives inclusion probabilities $P(S\subseteq X)=\det K_S$, not the probability $P(X=S)$ of observing exactly $S$. Those follow by inclusion-exclusion.

Take $K=\begin{pmatrix}3/5&1/5\\1/5&2/5\end{pmatrix}$. Its trace is $1$ and its determinant is $1/5$, so its eigenvalues $(1\pm 1/\sqrt5)/2$ lie strictly inside $(0,1)$. Principal minors give $P(101\in X)=3/5$, $P(-7\in X)=2/5$ and $P(\{101,-7\}\subseteq X)=1/5$. Hence

- $P(X=\{101,-7\})=1/5$,
- $P(X=\{101\})=3/5-1/5=2/5$,
- $P(X=\{-7\})=2/5-1/5=1/5$,
- $P(X=\varnothing)=1-3/5-2/5+1/5=1/5$.

The cardinality $N=|X|$ therefore has PMF $[1/5,3/5,1/5]$ and mean $\operatorname{tr}K=1$. The two states are dependent: $P(\text{both})=1/5$ differs from the product $6/25$ of the diagonal entries, so the diagonal is not a list of independent Bernoulli probabilities. Instead, $N$ is a sum of independent Bernoulli variables with the kernel's *eigenvalues* as success probabilities. That describes the count only, not which states are included.

```python
from fractions import Fraction

import numpy as np
from finitefree import DPPCountDistribution, PreparedDiscreteDPP

labels = [101, -7]  # matrix order: row/column 0 is state 101, 1 is state -7
K = np.array([[3, 1], [1, 2]]) / 5
assert np.all((np.linalg.eigvalsh(K) > 0) & (np.linalg.eigvalsh(K) < 1))

# Independent exact law from principal minors and inclusion-exclusion.
first, second = Fraction(3, 5), Fraction(2, 5)
both = first * second - Fraction(1, 5) ** 2
exact = {
    (): 1 - first - second + both,
    (101,): first - both,
    (-7,): second - both,
    (101, -7): both,
}
fifth = Fraction(1, 5)
assert exact == {(): fifth, (101,): 2 * fifth, (-7,): fifth, (101, -7): fifth}
exact_pmf = [sum(p for s, p in exact.items() if len(s) == n) for n in range(3)]
assert exact_pmf == [fifth, 3 * fifth, fifth]

prepared = PreparedDiscreteDPP(K, labels)
rng = np.random.default_rng(37)
for _ in range(5):
    sample = prepared.sample(rng=rng)
    assert len(set(sample)) == len(sample) and set(sample) <= set(labels)

law = DPPCountDistribution(K)  # the whole two-state space
expected = [float(p) for p in exact_pmf]
np.testing.assert_allclose(law.pmf(), expected, rtol=0, atol=1e-15)
assert abs(law.mean - 1) < 1e-15
assert abs(law.sf(0) - (expected[1] + expected[2])) < 1e-15  # strict: P(N > 0)
```

The sampled sets are checked for validity only; the exact sequence depends on the NumPy version. `DPPCountDistribution(K)` uses the kernel restricted to the region of interest, here the whole state space; `count_distribution_discrete(kernel, points)` builds the same law for chosen states of a `BaseKernel`. The upper tail is strict: `sf(k)` is $P(N>k)$, summed directly rather than computed as `1 - cdf(k)`. The count law is a float64 calculation from the kernel's eigenvalues; the fractions above are the independent reference.

For kernels, continuous gap approximations and sampling domains, see the [DPP contracts](api.md#orthogonal-polynomial-kernel-evaluation). Coefficient agreement and sampling smoke tests are diagnostics, not proofs of a distribution.
