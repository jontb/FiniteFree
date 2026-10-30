# P03 scope proposal: compact beta-Hermite sampling

**State: accepted and implemented as `sample_beta_hermite`; see the [API contract](api.md#compact-beta-hermite-sampling).** This is the concrete scope
record for [P03](packets/p03-beta-hermite.md), to carry into
[#43](https://github.com/jontb/FiniteFree/issues/43). Public names and
signatures below were accepted as written. P01/P02 are not prerequisites.

## Model and scale

Use the [Dumitriu–Edelman](https://arxiv.org/abs/math-ph/0206043) tridiagonal
model with real `beta > 0` and dimension `d >= 1`. With independent
`a_i ~ N(0, 2)` for `i=1..d` and `b_j ~ chi_{beta j}` for `j=1..d-1`, the
symmetric tridiagonal matrix `T` with diagonal `a` and off-diagonal
`b_{d-1},...,b_1` has joint eigenvalue density proportional to
`prod_{i<j} |lambda_i-lambda_j|**beta * exp(-sum lambda_i**2 / 4)`.

The library returns `H = scale * T / sqrt(beta d)`: diagonal entries
`N(0, 2 scale**2/(beta d))` and off-diagonal entries `scale * chi_{beta j}/sqrt(beta d)`.
At `beta=1` and `beta=2` the diagonal variances `2/d` and `1/d` equal those of
`sample_goe` and `sample_gue`. In general

\[
\mathbb{E}\,\frac{\operatorname{tr}H^2}{d}
=\frac{2d+\beta d(d-1)}{\beta d^2}
=1+\frac{2/\beta-1}{d},
\]

which matches the existing GOE, GUE and GSE (one eigenvalue per pair) samplers
at `beta=1,2,4`. A seeded check of 40,000 draws at `d=2,5` agreed with this
identity for all three within two standard errors; that is a consistency check,
not a law test.

Because entries are independent and centered, the expected characteristic
polynomial of the trailing `k x k` block satisfies
`p_k = x p_{k-1} - ((k-1)/d) p_{k-2}`, independent of beta, so it equals `gue_expected_poly(d)`: `x`, `x**2-1/2`, `x**3-x`,
`x**4-3/2*x**2+3/16` for `d=1..4`. This identity is the deterministic
expectation reference.

## Proposed public contract

| Proposed operation | Input and result |
| --- | --- |
| `sample_beta_hermite(d, beta, scale=1.0, *, rng=None)` | Positive integer `d`, finite real `beta > 0`, finite nonzero real `scale`. Return a new tuple `(diagonal, off_diagonal)` of writeable float64 arrays of lengths `d` and `d-1`. |

- Compact output only: O(d) storage, no hidden dense matrix or eigenvalue
  computation. The documentation shows `scipy.linalg.eigvalsh_tridiagonal`
  for eigenvalues and `np.diag(...)` for a dense matrix as explicit caller
  choices.
- Off-diagonal order is `b_{d-1},...,b_1` from the top, as in the reference.
  The spectrum does not depend on this order; the output contract does.
- `rng=None` uses the legacy global NumPy RNG, like the existing samplers; an
  explicit `Generator` is used for every draw and the global state is left
  untouched. Draw order is fixed and documented: `d` normals, then `d-1` chi
  variates generated as `sqrt(chisquare(beta j))`. Noninteger `beta` is
  supported through the chi-square shape.
- All arguments are validated before any random number is consumed. Existing
  GOE/GUE/GSE samplers, their formulas and draw order are unchanged.
- At `d=1` the result is one `N(0, 2 scale**2/beta)` value and an empty
  off-diagonal array.

## Acceptance

- Controlled draws (a stub generator returning fixed normals and chi-squares)
  check the exact construction, scaling, order and shapes.
- The variance identity above and the beta-independent expectation polynomial
  are checked against `gue_expected_poly(d)` exactly; the expected
  polynomial of the compact matrix is computed from its recurrence, not
  estimated from samples.
- `d=1` analytic law, noninteger beta, invalid `d`, `beta`, `scale` and `rng`
  types, ownership and RNG isolation (seeded reproducibility, global state
  unchanged with an explicit generator).
- A bounded seeded spectral comparison at beta 1, 2 and 4 against the dense
  samplers reports matched second and fourth moments through `matrix_moments`,
  with matrix-level standard errors and no must-pass p-value.

## Change and validation boundary

Likely changes: `finitefree/ensembles.py`, exports, proposed
`tests/test_beta_hermite.py`, `docs/api.md` and `CHANGELOG.md`. Run the focused
commands in the [P03 card](packets/p03-beta-hermite.md) in both modes, then
profile R. P04 reuses the compact-output and RNG conventions.

Excluded: replacing dense samplers, eigenvalue-returning variants, GPU paths,
a general backend framework and large law studies.
