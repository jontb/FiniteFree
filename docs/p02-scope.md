# P02 scope proposal: full inverse finite R

**State: accepted and implemented as `inverse_finite_r` and `matrix_finite_cumulants`; see the [API contract](api.md#finite-cumulant-reconstruction).** This is the concrete scope
record for [P02](packets/p02-inverse-finite-r.md), to carry into
[#42](https://github.com/jontb/FiniteFree/issues/42). The names and
signatures below were accepted as written. It builds on the
accepted [P01 contract](p01-scope.md) and the released `FiniteRTransform`
normalization.

## Mathematical target

For ambient dimension `d` and a complete sequence `kappa_1,...,kappa_d`, return
the unique monic degree-`d` polynomial whose `FiniteRTransform(p, order=d, d=d)`
equals that sequence. With the existing normalization

\[
\kappa_n^{(d)}=\frac{(-d)^{n-1}}{(n-1)!}\,c_n,
\]

where `c_n` are the classical cumulants of the normalized coefficients
`e~_0=1, e~_1, ..., e~_d`, the inverse sets `c_n = kappa_n (n-1)!/(-d)^(n-1)`,
recovers `e~_n = n! [z^n] exp(sum_n c_n z^n/n!)` and expands
`a_k = (-1)^k binom(d,k) e~_k`. Every step is exact over the rationals.

The reconstruction is formal. Arbitrary cumulants need not give a real-rooted
polynomial (for example `[0,-1]` at `d=2` gives `x**2+1/2`). The result keeps
lazy validation: it is never constructed with `assume_real_rooted=True`, and
no root property is asserted.

## Proposed public contract

| Proposed operation | Input and result |
| --- | --- |
| `inverse_finite_r(cumulants, *, d=None)` | A complete sequence of `d` rational cumulants (SymPy/FLINT/`Fraction`/integers, converted with existing helpers). Return a new monic `RealRootedPolynomial` of degree `d`. |
| `inverse_finite_r(cumulants, *, d=None, numerical=True)` | The same complete sequence of finite real floats. Return a new float64 descending coefficient array of length `d+1`, without forming a polynomial object. |

Prefer one exported function in `transforms.py` next to `FiniteRTransform`. The
name follows the P01 `snake_case` additions; `RealRootedPolynomial.from_finite_cumulants`
is the alternative if a constructor-style classmethod is preferred. Reuse the
existing exact series helper used by `additive_power`, without changing
`additive_power`, `FiniteRTransform` or their defaults.

Dimension and validation decisions proposed for acceptance:

- `d` defaults to `len(cumulants)` and, if supplied, must be a positive
  integer index equal to that length (P01 index rules: booleans and fractional
  values rejected). A shorter or longer sequence raises `ValueError`; missing
  cumulants are never filled with zeros. An empty sequence is invalid.
- An ambient `d` larger than a source degree is expressed by the cumulants
  themselves: `FiniteRTransform(p, order=d, d=d)` of a degree-`n` polynomial
  reconstructs `x**(d-n) p(x)` monic. No separate zero-root argument is added.
- Exact mode rejects floats, so float observations never become rational
  "exact" inputs by silent conversion. Callers who want exact rationals from
  floats convert explicitly. Nonrational symbolic values raise `ValueError`.
- Numerical mode uses float64 recurrences and raises `RuntimeError` for
  nonfinite coefficients. It returns coefficients, not roots, and carries no
  conditioning bound; high orders amplify input error.

## Empirical route and targets

The supported empirical input is a declared finite-cumulant estimate, for
example the finite cumulants of each matrix draw or their average. P02 adds
`matrix_finite_cumulants(matrix, *, d=None, paired=False)`, the float64
per-sample forward map. It shares `matrix_moments` validation and pairing, and
computes normalized elementary coefficients from the validated eigenvalues
after centering at the sample mean (restoring `kappa_1`), as the numerical
`FiniteRTransform` path does. It returns all `d` cumulants.

Three targets stay distinct and are labeled in the documentation:

| Target | For spectra `[1,3]` and `[3,5]` | Reconstruction |
| --- | --- | --- |
| Per-sample cumulants | `[2,2]` and `[4,2]` | `(x-1)(x-3)` and `(x-3)(x-5)` |
| Mean of per-sample cumulants | `[3,2]` | `(x-2)(x-4)` |
| Cumulants of the mean polynomial `(x-3)**2` | `[3,0]` | `(x-3)**2` |

Averaging cumulants and averaging coefficients do not commute, and neither
target identifies a unique matrix ensemble.

## Independent acceptance table

Computed by brute-force set partitions and a truncated SymPy logarithm in a
script that does not import `finitefree`:

| Input cumulants, `d` | Expected polynomial |
| --- | --- |
| `[2,2]`, `d=2` | `x**2-4*x+3` |
| `[1,3,9]`, `d=3` (roots `[0,0,3]`, the API example) | `x**3-3*x**2` |
| `[7/8,33/16,-81/32,-63/64]`, `d=4` (roots `[1/2,-1,2,2]`) | `x**4-7/2*x**3+3/2*x**2+4*x-2` |
| `[1,2,4,8]`, `d=4` (roots `[1,3]` in ambient dimension four) | `x**2*(x-1)*(x-3)` |
| `[3,2]`, `d=2` (mean per-sample cumulants above) | `(x-2)*(x-4)` |
| `[0,-1]`, `d=2` | `x**2+1/2`, not real-rooted |

Regressions should additionally cover nonmonic sources, shifts and dilations
(`kappa_1` shifts, `kappa_n` scales by `c**n`), additive convolution
(cumulants add), incomplete or overlong sequences, invalid dimensions, float
rejection in exact mode, numerical agreement with the exact path on rational
inputs, nonfinite numerical output, and empirical cumulants checked against
direct small-matrix determinant coefficients on the same draws. Round trips
supplement these values rather than replace them.

## Change and validation boundary

Likely changes: `finitefree/transforms.py`, `finitefree/ensembles.py` (the
matrix forward map beside `matrix_moments`), exports, proposed
`tests/test_inverse_finite_r.py` and additions to
`tests/test_empirical_transforms.py`, `docs/api.md`, `docs/tutorial.md` and
`CHANGELOG.md`. Run the focused commands in the
[P02 card](packets/p02-inverse-finite-r.md) in both modes, then profile R.

Excluded: prefix or partial-sequence containers, analytic inversion of the
infinite R-transform, positivity or real-rootedness inference, uncertainty
propagation beyond documenting it, and any change to `FiniteRTransform`.
