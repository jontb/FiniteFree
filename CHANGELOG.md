# Changelog

## Unreleased

- Exact rational conversion uses each Python/NumPy floating scalar's own integer ratio. NumPy extended-precision values are no longer narrowed through float64 before polynomial construction, affine transforms or exact matrix-pencil operations, including supplied-value symmetry checks. Numerical outputs retain their documented float64 limits.

- Generic companion/Aberth root paths normalize nonmonic coefficients exactly before float conversion. This fixes incorrect roots such as the numerical result for `2*x**2-6*x+4` and avoids common-scalar overflow/underflow without changing stored coefficients.

- Squarefree factors of degree at least 15 use certified Arb validation before the expensive integer PRS path. Degrees through 30 retain exact Sturm certification if Arb cannot obtain a certificate; small factors keep their existing exact path. This also reuses the isolated roots for numerical evaluation.

- Numerical real-root extraction rejects nonfinite final results before caching at every degree. Exact construction and real-rootedness certification remain available for roots outside the finite float64 range; representable values, multiplicities and read-only cache behavior are preserved.

- Arb validation-created root caches obey the read-only array contract, including cache reuse and later precision upgrades. Cross-batch regressions also check nonmonic input snapshots, direct projections and shifted numerical cumulants together.

- CI includes the advertised Python 3.9 minimum; documentation-tool annotations retain Python 3.9 compatibility.

- Linear symbolic roots are checked for realness instead of receiving an unconditional certificate. Certified numeric roots use their coefficient ratio for extraction and geometry; unknown realness/signs and nonnumeric/out-of-range numerical requests raise descriptive errors without poisoning caches.

- Polynomial construction copies mutable FLINT inputs; symbolic coefficient access returns independent arrays. Normalized coefficients and real/unitary root arrays are now read-only, with `.copy()` available for editing. T-transforms own their coefficient list instead of exposing the polynomial's cache.

- Normalized elementary coefficient sequences divide out the leading coefficient for `monic=False` inputs, making `e_0=1` and finite cumulants invariant under a nonzero scalar multiple. This also permits coefficient convolutions/reconstruction of nonmonic representations and corrects their additive powers. Stored coefficients and evaluations retain their scalar. Ratio-based S/T transforms preserve their existing values; symbolic common factors are simplified in the SymPy normalization path.

- README-to-MkDocs synchronization preserves fenced examples, rejects malformed details blocks before overwriting the index, and fails when the README is missing. Documentation tools now have strict typing, lint and regression checks in CI.

- Numerical finite cumulants center the requested coefficient prefix exactly before the Arb recurrence, preserving higher cumulants under large translations without a full polynomial shift.
- Finite T-transform interval selection preserves rational inputs near 0 and 1 and uses stored binary float values at grid boundaries. Degree-zero polynomials and invalid/nonfinite domain inputs raise `ValueError`. Use a rational input when an exact grid boundary is intended; rounded float products previously selected some neighboring intervals.

- Matrix-pencil rational methods preserve supplied integer/rational values before preparing float64 numerical views. This fixes loss of integers above `2**53` in characteristic polynomials, SLP derivatives and multivariate determinants. Constructors validate square, consistent, finite real rational inputs; symmetric pencils require symmetry on the supplied values without approximate tolerance. Reconstruct a pencil to change its entries.
- Cython modular evaluation reduces integers before int64 conversion, preventing overflow and repeated sparse interpolation retries for large entries.
- Arb real-rootedness validation retains its roots for evaluation, avoiding duplicate isolation and a subsequent companion-matrix pass. Higher working-precision reference requests refresh the cache.

- `EmpiricalComparison.verify_coefficients(alpha=0.05)` now checks every coefficient, including zero/tiny variance cases, and honors `alpha` using Bonferroni-adjusted two-sided Student-t bands. It replaces the fixed five-standard-error heuristic and no longer skips standard errors at or below `1e-10`. New keyword tolerances default to `rtol=1e-10, atol=0`; all sample variances are computed after coefficient scaling. Comparisons require at least two samples, finite Hermitian matrices of the matching dimension, and eigenvalue pairs for doubled-size samples.

- Orthogonal-polynomial kernels retain exact equality when selecting the Christoffel–Darboux diagonal formula; distinct integers above `2**53` are no longer conflated by float conversion. Nearby distinct floating points use the finite basis sum, preserving both coordinates and reducing quotient cancellation at additional evaluation cost. Diagonal and separated floating calls retain the fast formulas. An empty basis returns the zero kernel.

- Proper polynomial projections compute the leading `j+1` coefficients directly rather than constructing `d-j` intermediate derivatives. Exact coefficients and monic normalization are retained, including nonmonic inputs. Known Hermite projections preserve variance, center and numerical recurrence provenance. Full-degree projections retain object identity. Invalid noninteger or out-of-range dimensions raise `ValueError`.

- Finite free cumulants use `(-d)**(n-1) * c_n / (n-1)!`, following Definition 2.14 of Arizmendi et al. Earlier code multiplied by `(n-1)!`. For orders `n >= 3`, recompute saved cumulants or convert with `new = old / ((n-1)!)**2`. The first two cumulants are unchanged. The matching additive-power inverse is updated, preserving valid polynomial coefficients.
- Reconstructed normalized coefficient sequences and Jacobi/Laguerre polynomials outside proven orthogonality domains are validated lazily. Root extraction and positive-root domain checks reject complex-rooted inputs rather than accepting false certificates. Inability to certify raises `RuntimeError`.
- DPP sampling validates real symmetric correlation kernels, spectrum, shape and finiteness. Invalid kernels raise `ValueError`; numerical roundoff up to `1e-10` is allowed. Kernel subsets/permutations respect the requested state space.
- Stable tridiagonal root evaluation is available for classical orthogonal families, GUE/Wishart expectations, affine transforms and proven Hermite additive convolutions with `exact=False`. The default `exact=True` path is retained, and reference requests bypass approximate caches. Positive integer root powers preserve exact coefficients.

For stored cumulants from the earlier implementation:

```python
from math import factorial

saved_old_cumulants = [1, 3, 36]
migrated = [value / factorial(n - 1)**2
            for n, value in enumerate(saved_old_cumulants, start=1)]
assert migrated == [1, 3, 9]
```
