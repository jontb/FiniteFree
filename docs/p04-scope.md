# P04 scope proposal: compact beta-Laguerre sampling

**State: accepted and implemented as `sample_beta_laguerre`; see the [API contract](api.md#compact-beta-laguerre-sampling).** This is the concrete scope
record for [P04](packets/p04-beta-laguerre.md), to carry into
[#44](https://github.com/jontb/FiniteFree/issues/44). The names and
signatures below were accepted as written. It reuses the accepted
[P03](p03-scope.md) compact-output and RNG conventions.

## Model and parameter dictionary

Use the [Dumitriu–Edelman](https://arxiv.org/abs/math-ph/0206043) bidiagonal
model. For `beta > 0`, dimension `d >= 1` and a real shape `a`, the lower
bidiagonal `B` has independent entries

- diagonal `x_k ~ chi_{2a - beta(k-1)}` for `k = 1..d`;
- subdiagonal `y_k ~ chi_{beta(d-k)}` for `k = 1..d-1`.

The positive tridiagonal Gram matrix `L = B B^T` has eigenvalue density
proportional to `prod_{i<j} |lambda_i - lambda_j|**beta * prod lambda_i**(a-p) * exp(-sum lambda_i / 2)`
with `p = 1 + beta(d-1)/2`, for real `a > beta(d-1)/2`.

The library parameterizes the shape by the existing Wishart sample size: set
`a = beta n / 2` and normalize `W = scale * L / (beta n)`. Then

| Quantity | Value |
| --- | --- |
| Diagonal chi degrees of freedom | `beta (n - k + 1)`, `k = 1..d` |
| Subdiagonal chi degrees of freedom | `beta (d - k)`, `k = 1..d-1` |
| Density exponent `a - p` | `beta (n - d + 1)/2 - 1` |
| Admissible shape | real `n > d - 1` (every diagonal degree of freedom positive) |
| `E[W]` | `scale * I` |
| `E[tr W^2]/d` (scale one) | `1 + (d + 2/beta - 1)/n` |

At integer `n >= d` and `beta = 1, 2, 4`, `W` has the eigenvalue law of
`sample_wishart(d, n, beta)` (one eigenvalue per quaternionic pair at
`beta=4`), not its entry law. The second-moment row was derived from the
entry moments and matches independent real and complex Gaussian Wick
calculations at `beta=1, 2` for `d <= 3`. A seeded 200,000-draw check of the
existing `beta=4` sampler agreed within 1.2 standard errors. Integer `n < d`
(singular Wishart with zero eigenvalues) is outside the bidiagonal model's
domain and stays with `sample_wishart`.

## Expected characteristic polynomial

`L` is tridiagonal with `L_kk = x_k**2 + y_{k-1}**2` and
`L_{k,k-1} = x_{k-1} y_{k-1}`, so the characteristic recurrence
`p_k = (x - L_kk) p_{k-1} - L_{k,k-1}**2 p_{k-2}` has independent factors at
each step. Taking expectations, with `E x_k**2 = (n-k+1)/n` and
`E y_k**2 = (d-k)/n` after normalization, removes beta:

\[
\mathbb{E}\,p_k = \Bigl(x-\tfrac{n-k+1}{n}-\tfrac{d-k+1}{n}\Bigr)\mathbb{E}\,p_{k-1}
-\tfrac{(d-k+1)(n-k+2)}{n^2}\,\mathbb{E}\,p_{k-2}.
\]

For every real `n`, the result equals `n**-d d! (-1)**d L_d^{(n-d)}(n x)`, the
generalized Laguerre form already used by `wishart_expected_poly`. That was
checked symbolically in `n` for `d <= 5` with SymPy's `assoc_laguerre`, and
coefficient by coefficient against `wishart_expected_poly(d, n)` for integer
`n` in `d..d+3`. For rational `n` the coefficients are exact rationals, for
example `d=3, n=7/2`: `[1, -3, 15/7, -15/49]`. `wishart_expected_poly` keeps
its integer-only signature; tests use the SymPy form as the reference for
noninteger `n`.

## Proposed public contract

| Proposed operation | Input and result |
| --- | --- |
| `sample_beta_laguerre(d, n, beta, scale=1.0, *, rng=None)` | Positive integer `d`, finite real `n > d - 1`, finite real `beta > 0`, finite real `scale > 0`. Return a new tuple `(diagonal, subdiagonal)` of float64 arrays of lengths `d` and `d-1`: the factor `B * sqrt(scale/(beta n))`, so that `W = B B^T`. |

- **Returned object: the factor, not the Gram matrix or eigenvalues.** The
  documentation shows the explicit next steps: eigenvalues of `W` are the
  squared singular values of the factor, or the eigenvalues of its
  tridiagonal Gram matrix (diagonal `x_k**2 + y_{k-1}**2`, off-diagonal
  `x_k y_k`) through `scipy.linalg.eigvalsh_tridiagonal`. A dense `W` is built
  by the caller when wanted.
- `scale` must be positive because it enters the factor through a square root.
  `sample_wishart` and `wishart_expected_poly` are unchanged.
- RNG follows P03: `rng=None` uses the legacy global NumPy RNG; an explicit
  `Generator` never touches global state; other types raise `TypeError`. The
  draw order is fixed: `d` diagonal chi-squares with shapes
  `beta n, beta(n-1), ..., beta(n-d+1)`, then `d-1` subdiagonal chi-squares with
  shapes `beta(d-1), ..., beta`. No normal draws are needed.
- Every degree of freedom and `scale` is validated before any draw.
  `n <= d - 1` raises `ValueError` and names the bound. `n` close to the
  boundary is allowed; the last diagonal entry then concentrates near zero.
- At `d = 1` the result is one value whose square is
  `scale * chi2_{beta n} / (beta n)`, a gamma law with shape `beta n / 2` and
  scale `2 scale/(beta n)`, plus an empty subdiagonal.

## Empirical workflow

The documented example forms the Gram eigenvalues of each draw, applies
`matrix_moments` and `matrix_finite_cumulants` to the dense `W` of each draw,
and averages per-draw vectors. It shows these estimate moments and averaged
cumulants of the spectral law, while the beta-independent expectation
polynomial is a different target (P01/P02 aggregation rules). Eigenvalues are
positive almost surely, so no zero roots arise unless the caller passes an
ambient `d` larger than the matrix size.

## Acceptance

- Controlled draws (a stub generator returning fixed chi-squares) check the
  factor scaling, the draw order and degrees of freedom, and the Gram assembly
  against a separately formed dense product `B @ B.T`.
- Chi-square means as the stub's draws make `E[tr W]` and the expected
  characteristic polynomial exact deterministic checks (each is affine in the
  squared entries); the expected polynomial is compared with the SymPy
  `assoc_laguerre` form for rational `n`, including noninteger `n` and `n`
  near `d - 1`, and with `wishart_expected_poly` for integer `n`.
- The `d = 1` gamma law, noninteger beta and shape, the boundary and invalid
  parameters (`n <= d - 1`, nonpositive or nonfinite `beta`/`scale`, booleans,
  complex), ownership and RNG isolation.
- A bounded seeded comparison at beta 1, 2, 4 with `sample_wishart` spectra,
  using `paired=True` at beta 4, reports means and standard errors of the
  first two moments against the exact values above, with no must-pass
  p-value.

## Change and validation boundary

Likely changes: `finitefree/ensembles.py`, exports, proposed
`tests/test_beta_laguerre.py`, `docs/api.md` and `CHANGELOG.md`. Run the
focused commands in the [P04 card](packets/p04-beta-laguerre.md) in both
modes, then profile R.

Excluded: general-beta Jacobi (P07), singular `n < d` extensions, covariance
wrappers, eigenvalue-returning variants, a real-`n` extension of
`wishart_expected_poly` and large law studies.
