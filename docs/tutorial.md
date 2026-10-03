# Exact coefficients and numerical outputs

This tutorial uses the unreleased source described in the [API reference](api.md). Install that checkout using the [development guide](development.md).

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

## Exact matrix inputs

Pencils copy rational entries before making separate float64 views. Exact methods preserve values beyond float64's integer precision. Symmetry is checked on supplied values; exact determinant derivatives require a nonsingular evaluated matrix.

```python
from finitefree.hyperbolic import SymmetricMatrixPencil

large = 2**53 + 1
pencil = SymmetricMatrixPencil([[[large]]])
assert pencil.characteristic_polynomial([1])[0] == -large
assert pencil.evaluate([1])[0, 0] == float(large)
```

For kernels, continuous gap approximations and sampling domains, see the [DPP contracts](api.md#orthogonal-polynomial-kernel-evaluation). Coefficient agreement and sampling smoke tests are diagnostics, not proofs of a distribution.
