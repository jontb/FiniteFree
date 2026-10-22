# P01 scope proposal: exact and empirical moments

**State: proposed for review; no implementation.** Selected by the
[P00 audit](p00-baseline-report.md). This is the concrete scope record for
[P01](packets/p01-exact-moments.md), to carry into the required API discussion.
Public names and signatures below are proposals, not available imports.

The public feature/API discussion is now
[#41](https://github.com/jontb/FiniteFree/issues/41). Its creation does not mark
these proposed signatures accepted or the feature implemented.

## Mathematical target

For a nonzero degree-n polynomial with roots `lambda_1,...,lambda_n`, use a
positive integer ambient dimension `d >= n`. Embed `d-n` additional zero roots.
Define `m_0 = 1` and, for `k >= 1`,

\[
m_k^{(d)}(p)=\frac{1}{d}\sum_{j=1}^{n}\lambda_j^k.
\]

Recover these numbers by Newton identities on the monic coefficient sequence;
do not numerically extract roots. Exact rational coefficients produce exact
rational moments, invariant under a nonzero common coefficient scalar.
This algebraic operation does not certify real-rootedness or positivity.
Orders above the degree remain meaningful and generally nonzero, unlike the
current finite-R convention above ambient dimension.

For one Hermitian sample M of effective dimension n, the empirical counterpart
has the same target and zero embedding. Positive-order moments are `tr(M**k)/d`
in the ordinary representation. For a physical `2n` matrix with duplicated
eigenvalues and explicit paired interpretation, use `tr(M**k)/(2*d)`.
Set `m_0=1` in both cases, including ambient zero roots. `scale` acts on the
actual sampled matrix; its k-th moment scales by `scale**k`.

## Proposed public contract

| Proposed operation | Input and result |
| --- | --- |
| `root_moments(p, order, *, d=None)` | Existing `RealRootedPolynomial`, with rational coefficient conversion through existing helpers. Return a newly owned list of exact SymPy rational values `m_0,...,m_order`. |
| `matrix_moments(matrix, order, *, d=None, paired=False)` | One finite, nonempty, square numerical Hermitian matrix. Return a newly owned float64 array `m_0,...,m_order`, computed without forming its characteristic polynomial. |

Prefer these two small functions exported from `finitefree` over a new general
transform class, sample container or backend framework. Put exact logic in
`transforms.py`; put the matrix adapter in an agreed existing module, factoring
only the small shared validation needed by future empirical routes. Preserve
the current `FiniteRTransform`, polynomial storage and comparison defaults.

Dimension and validation decisions proposed for acceptance:

- `order` is a nonnegative integer index; `d`, if supplied, is a positive
  integer index. Accept objects implementing the integer-index protocol;
  reject booleans, fractional values and negatives explicitly. Do not tighten
  unrelated legacy APIs as part of this packet.
- Default `d` is the polynomial degree or effective matrix dimension. Reject
  `d` smaller than that dimension. A nonzero constant polynomial requires an
  explicit positive `d`, and returns `[1,0,...,0]`. The zero polynomial remains
  invalid. Empty matrices are outside the initial empirical contract.
- Matrix input may be real or complex Hermitian. Own any working arrays and do
  not mutate the supplied matrix. Check finite entries, square shape and the
  current empirical Hermitian tolerance (`rtol=0`, absolute tolerance
  `1e-10*max(abs(M))`). Document that this is numerical validation.
- `paired=True` is explicit: require even physical dimension and verify sorted
  adjacent eigenvalue pairing with the current scale-relative tolerance before
  counting one eigenvalue per pair. Default `paired=False` uses all physical
  eigenvalues. An even matrix size alone never implies quaternionic pairing.
- Use float64/complex128 numerical evaluation and reject nonfinite computed
  results with a descriptive error. No exactness, enclosure or high-order
  stability claim follows from floating matrix input. Preserve numerical error
  separately from sampling uncertainty. Do not silently clip, repair, filter
  failed draws or form exact rationals and label them ensemble information.

The empirical route can use a Hermitian eigensolver and spectral power sums,
with direct matrix-power traces as its independent low-dimensional reference.
Do not route it through the current `_sample_observations` characteristic
coefficient calculation: coefficient overflow must not reject a representable
low-order moment. No cache expansion is needed for this initial implementation.

## Aggregation and statistical interpretation

The functions return a single polynomial's moments or one matrix's moments.
For independent draws, average the per-matrix vectors to estimate moments of
the mean empirical spectral measure. At a fixed dimension this agrees with
integrating powers against the estimated mean spectral measure. It generally
differs from root moments of the mean characteristic polynomial.

The comparison example will use 16 actual `sample_gue(4, rng=...)` draws with
seed 862, evaluating both the proposed matrix route and direct normalized
matrix-power traces on each same draw through order four. Show sample mean,
sample standard deviation (`ddof=1`) and standard error `std/sqrt(16)` at the
matrix level. These numbers are descriptive estimates from independent
matrices, not deterministic accuracy bounds. Use deterministic identities for
pass/fail assertions; do not require a random confidence interval to contain an
ensemble expectation. Do not assert equality to moments of
`gue_expected_poly(4)`.

No retained/streaming comparison extension is included. If a later packet
offers both forms, it must accumulate the same declared per-sample target.
An existing streaming mean coefficient vector cannot recover unrecorded
nonlinear sample statistics.

## Independent acceptance table

P00 checked the fixed values below using rational root powers and exact
matrix-power traces in `experiments/packets/P00/run-001/oracle-check.py`.
These checks validate the proposed examples, not an unimplemented API.

| Input | Independent expected output through order four |
| --- | --- |
| `(x-1)(x-3)`, `d=2` | `[1,2,5,14,41]` |
| Same polynomial, `d=4`, embedded roots `[0,0,1,3]` | `[1,1,5/2,7,41/2]` |
| Complex Hermitian `[[2,i],[-i,2]]`, `d=2` | `[1,2,5,14,41]` by exact powers/traces |
| Physical diagonal `[1,1,3,3]`, paired, effective `d=2` | `[1,2,5,14,41]`; use denominator four in the physical trace |
| Same paired matrix, ambient `d=4` | `[1,1,5/2,7,41/2]`; use denominator eight |
| Two samples with spectra `[1,3]` and `[3,5]` | Mean second spectral moment `11`; mean polynomial `(x-3)**2` has root second moment `9` |

The aggregation case also has per-draw current finite cumulants `[2,2]` and
`[4,2]`, whereas the mean polynomial has `[3,0]`. Keep this target distinction
visible for the subsequent P02 forward/inverse implementation.

P01 regressions should additionally cover nonmonic input, rational roots,
negative/repeated/zero roots, order zero and orders above degree; shifts and
dilations against direct root sums; constant polynomials with ambient zeros;
invalid index/dimension/matrix domains; paired mismatch; noncontiguous input;
input/output ownership; representable low-order moments despite coefficient
overflow; and explicit nonfinite-output failures. Round trips supplement these
independent expectations rather than replace them.

## Change and validation boundary

Likely changes: `finitefree/transforms.py`, public exports and one small
empirical adapter; proposed `tests/test_root_moments.py` and
`tests/test_empirical_transforms.py`; an executable API/tutorial example and
`CHANGELOG.md`. Touch `core.py` only for a justified small coefficient-access
change. Use existing rational normalization and test conventions.

Run the focused commands in the [P01 card](packets/p01-exact-moments.md), then
the full contributor profile R and documentation profile D. Confirm the new
tests were actually collected in both compiled and flag-fallback modes.
Precision, RNG, ownership and legacy transform regressions remain required.

This scope excludes inverse finite-R, empirical Cauchy/S/T/symmetric-S adapters,
new samplers, arbitrary-precision empirical moments, general streaming APIs,
certification guarantees and performance studies. Their assigned packets are
recorded in the [P00 inventory](p00-baseline-report.md#empirical-transform-inventory-and-assigned-gaps).
The next decision is acceptance of these names, dimensions, pairing and output
types in the required API discussion before P01 runtime work begins.
