# Changelog

## Unreleased

- Add `sample_beta_hermite` for the compact Dumitriu-Edelman tridiagonal model
  at any positive real beta, returning diagonal and off-diagonal arrays. Its
  scale matches the GOE, GUE and GSE samplers at beta 1, 2 and 4, and its
  expected characteristic polynomial is `gue_expected_poly(d)`. Existing
  samplers, formulas and draw order are unchanged.
- Add `root_moments` for exact normalized root moments with ambient zero roots,
  computed from rational coefficients by Newton identities without root
  validation. Add `matrix_moments` for the float64 moments of one Hermitian
  sample, with explicit paired representations. Existing transforms,
  comparisons and defaults are unchanged.
- Cache the first successful prepared-quadrature count law for repeated count
  and gap queries. Returned laws retain independent arrays and metadata;
  spectrum validation remains lazy and failed computations are retried.
- Fix `RealRootedPolynomial.evaluate_roots_float64(parallel=True, exact=False)`
  returning unconverged Aberth–Ehrlich real parts. Ill-conditioned generic
  coefficients, such as roots 1..32, previously produced errors above 20 on
  both the CuPy and NumPy seekers. Seeker results are now accepted only after
  convergence and the companion path's real-root test; otherwise the call
  continues with the existing companion and Arb fallbacks. Converged results,
  defaults and signatures are unchanged.
- Stop certifying real roots that do not exist. `hahn_polynomial` now trusts
  real-rootedness only for `alpha > -1` and `beta > -1` (other parameters are
  validated lazily, as for Jacobi) and rejects non-integer `N`, which was
  silently truncated. `wishart_expected_poly` requires integers `d >= 0` and
  `n >= 1`. Polynomials derived from a `UnitaryPolynomial` (`derivative`,
  `shift`, `dilation`, `projection`, `square_roots_map` and copies made by
  empirical comparisons or DPP kernels) no longer inherit its unit-circle
  trust as a real-root certificate, and `verify_root_interlacing` raises
  `ValueError` when real-rootedness does not hold. Previously these cases
  returned `True` and reported complex roots as repeated real parts.
- Speed up exact cumulant and transform paths without changing results.
  `FiniteRTransform` (exact) and `additive_power` take classical cumulants from
  a FLINT series logarithm instead of a quadratic Python recurrence (order 400:
  1.07 s to 0.06 s; `additive_power(3/7)` at degree 250: 0.69 s to 0.026 s).
  `FiniteCauchyTransform` reads coefficients once (degree 400: 1.19 s to
  0.07 s). `from_roots` caches correctly rounded roots as exact, so the default
  `exact=True` call no longer isolates known roots again (degree 200: 0.76 s
  to under 1 ms). Real empirical samples keep the real symmetric eigensolver
  (GOE, degree 500: 39.5 ms to 15.4 ms per sample). Timings are single-thread
  measurements on one Linux machine; outputs were checked identical.
- Fix exact-arithmetic utility defects. `PrecisionContext` restored the
  precision seen at construction rather than at entry, so a precision change in
  between, or nested reuse, was lost on exit. `flint_to_float` rounded twice and
  was one ulp off for about a third of large ratios; it now rounds once. The
  compiled `modular_det` gave wrong results for negative entries and overflowed
  above int64, the compiled `crt` overflowed for primes above about 3.04e9, and
  `PYFFP_DISABLE_CYTHON=1` did not disable the compiled `modular_det`; entries
  are now reduced exactly before narrowing, large primes use Python arithmetic,
  and the flag is honoured. Internal callers did not reach these inputs. The
  exact Sturm fallback used when Arb cannot certify degrees 15-30 raised or
  miscounted for abnormal remainder sequences; it now uses a sign-correct
  integer Sturm sequence and raises `ValueError` for non-real-rooted inputs.
- Fix float straight-line-program derivatives, empty pencils and operation
  validation. Float determinant gradients and Hessians switched to a
  pseudoinverse whenever `|det| < 1e-15`, returning wrong derivatives at
  nonsingular points such as `diag(x1, x2)` at `(1, 1e-16)`; they now use LU
  solves with log-scaled products and, as prepared pencils do, raise
  `numpy.linalg.LinAlgError` at an exactly singular point instead of returning
  zeros. An empty 0x0 pencil made the interpolated constructor loop forever
  with the compiled extension; it and `diagonal_specialization` now return 1,
  per-prime workers retry only singular modular systems, and repeated failures
  stop after eight primes. `StraightLineProgram` validates operation codes,
  arity and operand indices at construction, recognises any sequence equal to
  `["det"]`, and raises `NotImplementedError` for Hessians of generic programs
  instead of returning the determinant Hessian.
- Evaluate prepared orthogonal-kernel bases by an exactly verified three-term
  recurrence. `prepare_numeric()` used Horner's rule on monomial coefficients
  for every family except Hermite, so Legendre bases had relative error 2.6e-7
  at rank 30 and were wrong at rank 50, and count laws raised "eigenvalues must
  lie in [0, 1]" from rank 30 on. Any basis whose monic recurrence and norm
  ratios check exactly in rational arithmetic (shifted, scaled or signed
  Hermite, Legendre, Laguerre, Jacobi) now uses the orthonormal float
  recurrence (errors near 1e-14 through rank 50); other bases keep Horner. The
  Hermite fast path also dropped the sign of sign-flipped basis polynomials in
  `basis()`; every recurrence column now takes the sign of its supplied
  leading coefficient.
- Reject repeated and negative discrete DPP states. `gap_probability_discrete`
  raised no error for repeated points and returned the determinant of a
  duplicated principal matrix (`gap([0, 0])` gave 0 instead of `gap([0])`).
  `DiscreteFiniteKernel` let negative indices wrap to the end of list and
  ndarray matrices, so `[4, -1]` silently meant state 4 twice in count laws,
  gaps and sampling. Both now raise (`ValueError` for repeated points,
  `IndexError` for negative indices), and the kernel stores a copy of its
  matrix so later edits to the caller's array cannot change it.

## 0.4.1

### Package descriptions and documentation

- Replace the broad exactness-focused package summary with reusable numerical
  workflows and reproducible sampling. Use a concise PyPI-specific description
  with working absolute documentation links and explicit precision/performance
  limits; add project documentation, source, issue and changelog metadata links.
- Include the refreshed getting-started, API, setup and validation guidance in
  the source release. This patch changes metadata and documentation only:
  runtime source, defaults, mathematical normalization and dependencies match
  0.4.0. Existing callers need no migration.

## 0.4.0

### Optional numerical workflows and maintenance

- Add opt-in prepared orthogonal kernels with batched normalized-basis evaluation
  and reusable finite-interval quadrature. Exact scalar evaluation and existing
  gap-probability defaults remain unchanged.
- Expose finite-rank DPP count PMFs, CDFs, directly summed upper tails, mean and
  variance. Continuous laws are quadrature estimates with visible spectrum/mass
  diagnostics, not certified probability bounds.
- Add incremental empirical coefficient moments and interval-count histograms
  with storage independent of the number of samples. The existing empirical
  comparison continues to retain samples by default.

- Add optional keyword-only NumPy `Generator` arguments to the Gaussian,
  Wishart and Haar samplers. Omitting `rng` or passing `None` preserves the
  legacy global RNG, draw order, matrix formulas and normalization.
- Generate both interactive asset families in the full contributor check before
  strict MkDocs. Run representative adversarial fault audits with clean controls.
- Replace recursive full-suite fallback tests with focused compiled, flag-fallback
  and missing-extension subprocess probes. Installed-wheel checks explicitly run
  both compiled and flag-fallback suites, without a whole-suite 30-second limit.
- Validate release tag/version agreement and prerelease routing before builds;
  inspect archive metadata, runtime source and native inventories and preserve
  SHA-256 manifests. Verify downloaded upload bytes against the reviewed manifest.
  Manual dispatch still defaults to TestPyPI. Runtime dependency floors and
  existing required CI check names are unchanged.
- Regenerate Cython C for each build target, preventing C generated against newer
  NumPy declarations from breaking a later build against the minimum headers.

### Migration from 0.3.0

- Existing calls need no migration. Ensemble samplers still use the global NumPy
  RNG when `rng` is omitted or `None`; pass a `numpy.random.Generator` explicitly
  to use an independent stream. Matrix formulas and normalizations are unchanged.
- Prepared/batched orthogonal kernels and DPP count laws are optional float64
  APIs. Exact scalar kernels and existing gap-probability APIs retain their
  behavior. Supply the kernel's reference density for continuous restrictions;
  the default density is 1. Quadrature results and float64 tail probabilities
  are numerical estimates, not certified bounds.
- `EmpiricalComparison` continues to retain samples. Choose
  `StreamingEmpiricalComparison` for bounded-storage coefficient/count summaries;
  it retains no matrices or eigenvalues for later plots. Interval counts use
  closed intervals and sample variances use `ddof=1`. Both comparison classes
  retain the documented independence and t-band assumptions; online roundoff
  can differ from retained two-pass statistics near a decision threshold.
- Python 3.10 and all runtime dependency floors are unchanged from 0.3.0.
  No GPU performance or certified rare-tail guarantees are added.

## 0.3.0

### Prepared numerical pencils and spectral workflows

- Add `PreparedMatrixPencil`, `PencilFactorization` and
  `SymmetricMatrixPencil.prepare_numeric()` for owned float64 coefficient
  snapshots, batched evaluation, reusable pivoted LU factors, slogdet,
  log-absolute-determinant gradients and Hessian-vector products. NumPy is the
  default; the optional CuPy backend uses explicit resident device arrays.
  Exact FLINT, polynomial and SLP methods retain their existing behavior.
- Verify shifted/scaled Hermite bases and their norm ratios before using a
  normalized numerical recurrence. Preserve measure mass, signed basis scales,
  generic fallbacks and public kernel signatures.
- Add integrated GUE spectral and interval-count concentration examples with
  exact deterministic references, recorded Monte Carlo uncertainty and retained
  discrepancies. Quadrature refinements are numerical diagnostics, not certified
  rare-tail bounds.
- Add standalone interactive geometry and spectral dashboards, including bounded
  compound-Wishart/free-lognormal and continuous-gap comparisons. Precompute the
  data outside the browser and exercise controls and responsive layouts in CI.

### Correctness and compatibility

- Matrix-pencil constructors reject complex entries before rational conversion,
  including Python/NumPy complex scalars with zero imaginary part. Callers with
  mathematically real complex arrays must explicitly validate and extract their
  real values. Valid real/rational inputs retain their supplied-value semantics.

- Diagonal matrix-pencil specialization now reconstructs until the CRT modulus
  exceeds twice a proved coefficient bound. Repeated modular agreement can no
  longer hide large nonzero coefficients.
- Specialization weights and offsets preserve exact rational and supplied binary
  floating values, including NumPy extended precision. Nonfinite, complex and
  irrational parameters raise `ValueError` before modular evaluation.
- Discrete DPP sampling rejects complex entries in callable kernels and object
  arrays before float conversion or random draws, consistently with complex
  numerical arrays. Valid real kernels retain the existing sampling algorithm.
- Exclude local `wheelhouse/` artifacts from source distributions, preventing
  stale wheels from being bundled in a new release archive.

### Tests and contributor tooling

- Integrate adversarial test consolidation with the complete GUE workflow:
  965 collected tests become 419 on the local Python 3.12 configuration, with
  identical executed runtime lines and branches in compiled and fallback modes.
  All eight representative injected faults are detected, including stale HVP
  directions and a deterministic Haar sampler missed by the original suite.
  This preserves input cases; it does not imply proportional runtime savings.
- Add a portable contributor check runner, contribution/issue/PR guidance and a
  code map. Python 3.10 and the existing runtime dependency floors are unchanged.
- Genuine RTX 3050/CuPy correctness tests cover singular and mixed batches.
  GPU latency, total device memory and crossover benchmarks remain unmeasured;
  no GPU speed claim is made.

## 0.2.0 — 2026-10-07

Release source: [v0.2.0](https://github.com/jontb/FiniteFree/tree/v0.2.0). Distribution: [PyPI 0.2.0](https://pypi.org/project/finitefree/0.2.0/). See the [release process](docs/release.md) for validation and publishing.

### Multivariate numerical and context workflows

- Reduce multivariate numerical work with cached nonzero exponent coordinates, reused monomial/component buffers and bounded 8192-point blocks with contiguous coordinate columns. Derivative power storage no longer grows with the full batch; output storage, coordinate conversion/validation and lazy exact caches retain their own costs. Small/simple cases can incur extra block overhead or temporary storage.
- Fill constant numerical results directly after coordinate/coefficient validation, including gradients of linear polynomials and Hessians of quadratics. This avoids unnecessary coordinate packing and power/buffer setup while preserving owned shapes and finite-input checks.
- Use native sparse context projection for validated reordering and cross-context symbol substitutions; construct zero results and target generators without SymPy reconstruction. Same-context substitutions use composition because python-flint 0.9 projection ignores explicit maps when the context is unchanged. Independent regressions cover swaps, merged variables, large exact exponents and strided batches.
- Add `scripts/benchmark_multivariate_scaling.py` for pinned Git baseline comparisons, sparse/dense and degree/bit-size scaling, cold/warm timings, interleaved samples, allocation peaks, fresh-process RSS and profiles. Numerical references independently differentiate rational monomials; exact contexts are checked against SymPy.
- Add `MultivariatePolynomial.reorder_variables` for explicit expression-preserving context permutations and `substitute` for simultaneous rational scalar/polynomial composition into an explicit target context. Symbol identity, ordering, exact scalar precision and owned results are preserved; no stability or hyperbolicity certificate is inferred.
- Add `gradient_float64` and `hessian_float64` with the existing `(..., variable_count)` batch input contract. Exact differentiation precedes coefficient conversion; powers are reused within each call, Hessians evaluate only the upper triangle, and zero coordinates/singular determinant points are supported. Output arrays are independent float64 snapshots, with explicit input, overflow, underflow and rounding boundaries.
- Reject NumPy complex scalars hidden in object arrays in all multivariate numerical batch APIs instead of allowing a lossy real cast.

### Migration from 0.1.0

- Python 3.10 and `python-flint>=0.9.0` are now required. An actual Python 3.9 / FLINT 0.6 test reproduced the missing rational multivariate API. Stable FLINT 0.7/0.8 require Python 3.11; FLINT 0.9 supports Python 3.10 and supplies the needed API. The remaining runtime floors are unchanged and pinned in the minimum-dependency CI job.
- `SymmetricFiniteSTransform` keeps the existing even-coefficient ratio by default. Opt into Definition 8.1's positive-imaginary complex transform with `convention="standard"`; its domain requires positive degree and real-rootedness.
- Orthogonal kernels now snapshot their basis, norms and leading coefficients. Public exports are independent copies; reconstruct the kernel rather than mutating an exported basis to change it.
- Recompute saved cumulants of orders 3 and higher with the corrected normalization. For the same normalized input and ambient dimension, an old cumulant can be converted with `new = old / ((n-1)!)**2`. This factor alone does not repair separately incorrect nonmonic normalization or precision already lost in saved data.
- Cached normalized coefficients and real/unitary root arrays are read-only; use `.copy()` before editing. Constructors now snapshot polynomial and matrix inputs.
- Lazy geometry checks and nonfinite root errors are intentional contracts. `exact=True` selects a high-precision path with numerical fallbacks; outputs remain float64, not certified intervals or exact algebraic values.
- Empirical coefficient verification now honors `alpha` and checks every coefficient. A passing diagnostic is not proof of a sample distribution.
- Exact T-grid boundaries require rational inputs; floats are evaluated at their stored binary positions.

### Multivariate rational polynomial foundations

- Repair exact public evaluation on python-flint 0.9.0 using its positional callable interface.
- Convert native FLINT exponents to Python integers at symbolic exports and normalized coefficient keys, preserving exact integer powers with SymPy 1.12. Minimum-dependency tests compare rational values explicitly across the two backends.
- Own native polynomial inputs, exported native copies, coefficient maps and variable lists. Reject native context/name order mismatches and duplicate variable names rather than silently relabeling coordinates.
- Add sparse coefficient construction/roundtrip, rational addition/subtraction/multiplication and nonnegative integer powers, exact gradient/Hessian polynomials, and scalar-preserving line restriction with lazy univariate geometry certification.
- Add real float64 batch evaluation with finite-input/coefficient/result checks and documented cancellation/underflow limits.
- Reject negative/fractional/boolean derivative orders and nonhomogeneous multinomial normalization. These previously allowed silent no-op or mathematically undefined results.
- Native exports now return independent copies; callers modifying the previously exposed live object must reconstruct a new polynomial explicitly.
- Modular determinant reconstruction now uses a proved coefficient-size bound to determine the necessary CRT modulus. Consecutive agreement previously allowed large nonzero coefficients divisible by the early primes to be reconstructed as zero.
- Sparse determinant construction independently verifies randomized discovery with an exact bounded Kronecker encoding. Balanced-base recovery repairs omitted or incorrect terms and provides a fallback after eight failed prime fields. The new keyword-only `max_verification_bits=1_000_000` raises `ValueError` before discovery when verification exceeds its conservative integer-size bound; callers needing larger encodings must opt in explicitly or choose another constructor.

### Documentation and release validation

- README/API describe the 0.2.0 release and its matching PyPI installation. Installation, development checks, tutorial, precision/evaluation boundaries, determinant derivative limits and implementation-dependent complexity are documented consistently.
- README is included as the package long description. Static version 0.2.0, wheel/sdist builds and the release checklist are aligned with the tagged release.
- The 12-job supported-platform CI matrix is supplemented by actual pinned minimum-runtime tests on Python 3.10 and release-candidate artifact builds.

### Implemented changes

- Add `PreparedDiscreteDPP`, which validates and snapshots a correlation kernel and state order, decomposes it once, and reuses the eigensystem for independent HKPV draws. Both prepared and raw samplers accept an optional NumPy `Generator`; default calls preserve the legacy global RNG. Tests compare seeded outputs and independent principal-minor probabilities for small projection and nonprojection kernels.

- Floating kernels with exactly verified probabilists' Hermite coefficients, norms `j!` and leading coefficients 1 use a normalized three-term recurrence and `math.fsum` for all coordinate pairs. Generic bases retain the established Christoffel–Darboux and nearby finite-sum paths. Construction snapshots inputs; finite/range errors are explicit. Independent exact Hermite references test nearby, diagonal and separated points through degree 80.

- Add the standard complex symmetric-S convention while retaining the default ratio. Exact expressions take the positive-imaginary branch; numerical square roots precede float narrowing and reject nonzero outputs outside complex128 range.

- Truncated cumulant requests normalize only the requested coefficient prefix. The native-dimension prefix cache is bounded by the polynomial degree and never masquerades as complete coefficients; ambient dimensions and nonmonic leading scalars retain exact normalization.

- Exact rational conversion uses each Python/NumPy floating scalar's own integer ratio. NumPy extended-precision values are no longer narrowed through float64 before polynomial construction, affine transforms or exact matrix-pencil operations, including supplied-value symmetry checks. Numerical outputs retain their documented float64 limits.

- Generic companion/Aberth root paths normalize nonmonic coefficients exactly before float conversion. This fixes incorrect roots such as the numerical result for `2*x**2-6*x+4` and avoids common-scalar overflow/underflow without changing stored coefficients.

- Squarefree factors of degree at least 15 use certified Arb validation before the expensive integer PRS path. Degrees through 30 retain exact Sturm certification if Arb cannot obtain a certificate; small factors keep their existing exact path. This also reuses the isolated roots for numerical evaluation.

- Numerical real-root extraction rejects nonfinite final results before caching at every degree. Exact construction and real-rootedness certification remain available for roots outside the finite float64 range; representable values, multiplicities and read-only cache behavior are preserved.

- Arb validation-created root caches obey the read-only array contract, including cache reuse and later precision upgrades. Cross-batch regressions also check nonmonic input snapshots, direct projections and shifted numerical cumulants together.

- Documentation-tool annotations avoid unsupported evaluation of newer annotation syntax. The runtime support floor is now Python 3.10 after testing the original dependency declaration.

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

- Orthogonal-polynomial kernels retain exact equality when selecting the Christoffel–Darboux diagonal formula; distinct integers above `2**53` are no longer conflated by float conversion. Nearby distinct floating points use the finite basis sum, preserving both coordinates. Generic diagonal and separated calls retain the fast formulas; verified Hermite inputs use the recurrence described above. An empty basis returns the zero kernel.

- Proper polynomial projections compute the leading `j+1` coefficients directly rather than constructing `d-j` intermediate derivatives. Exact coefficients and monic normalization are retained, including nonmonic inputs. Known Hermite projections preserve variance, center and numerical recurrence provenance. Full-degree projections retain object identity. Invalid noninteger or out-of-range dimensions raise `ValueError`.

- Finite free cumulants use `(-d)**(n-1) * c_n / (n-1)!`, following Definition 2.14 of Arizmendi et al. Earlier code multiplied by `(n-1)!`. For orders `n >= 3`, recompute saved cumulants or convert with `new = old / ((n-1)!)**2`. The first two cumulants are unchanged. The matching additive-power inverse is updated, preserving valid polynomial coefficients.
- Reconstructed normalized coefficient sequences and Jacobi/Laguerre polynomials outside proven orthogonality domains are validated lazily. Root extraction and positive-root domain checks reject complex-rooted inputs rather than accepting false certificates. Inability to certify raises `RuntimeError`.
- DPP sampling validates real symmetric correlation kernels, spectrum, shape and finiteness. Invalid kernels raise `ValueError`; numerical roundoff up to `1e-10` is allowed. Kernel subsets/permutations respect the requested state space.
- Stable tridiagonal root evaluation is available for classical orthogonal families, GUE/Wishart expectations, affine transforms and proven Hermite additive convolutions with `exact=False`. The default `exact=True` path is retained, and reference requests bypass approximate caches. Positive integer root powers preserve exact coefficients.

For stored cumulants from the earlier implementation:

```python
from math import factorial
from fractions import Fraction

saved_old_cumulants = [1, 3, 36]
migrated = [Fraction(value, factorial(n - 1)**2)
            for n, value in enumerate(saved_old_cumulants, start=1)]
assert migrated == [1, 3, 9]
```

## 0.1.0 — 2026-06-19

Published on PyPI from `v0.1.0`. Requires Python ≥3.9. Its original cumulant formula multiplied by `(n-1)!` and its contracts predate the development fixes above. Published Python 3.9 wheels do not validate runtime compatibility of subsequent local changes.
