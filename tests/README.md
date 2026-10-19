# Test contracts

Run both implementations from the repository root:

```sh
PYTHONPATH=. python -m pytest --import-mode=importlib tests/ scripts/tests/
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/ scripts/tests/
```

Related data cases share a contract test and explicit loops. `case(...)` labels
failures with their inputs and preserves the original exception as the cause;
skips and interrupts propagate. Use `pytest -k contract_name` to rerun a family.
API variants retain separate reports where useful. Tests requiring fresh
fixtures, monkeypatches, or platform skips stay independently parametrized.

CI and `scripts/check.py --full` run compiled and flag-fallback suites explicitly;
the installed-wheel helper also runs both modes outside the checkout. No test
recursively runs a full pytest suite. A small subprocess probe verifies compiled
selection, the documented fallback flag, and actual native-import failure.

Consolidation reduces collected pytest items without dropping the original
numerical cases or expecting proportional runtime savings. Keep independent
oracles and edge cases when changing the suite:

- Exact rational cumulants use partition enumeration and a truncated formal
  logarithm in `_cumulants.py`. Fixed-value, nonmonic, prefix, and transformation
  checks share these assertions; additivity alone cannot distinguish two
  equally wrong answers.
- DPP samples are checked against every exact subset probability, including a
  dense three-state kernel. Raw and prepared samplers share the law check while
  retaining their seeded trials, legacy RNG coverage, and probability bounds.
- Prepared-pencil derivatives use independent symbolic determinants, extreme
  coefficient scales, strided batches, varied block sizes, and cache reuse.
- Prepared kernels use exact rational basis sums with signed/scaled/shifted
  families. Count laws use exhaustive Bernoulli enumeration and an analytically
  integrated rank-two Gram matrix, including tails below CDF resolution.
- Streaming coefficient bands use closed-form t quantiles at extreme scales;
  moments/counts agree with retained samples across update order and rescaling.
  Invalid updates preserve state and stored array shapes do not grow with samples.
- Root, CRT, matrix-input, and derivative cases retain range, ownership, and
  invalid-input checks alongside their exact or analytic references.
- The root-cache lifecycle checks ownership and multiplicity while counting
  isolations across reuse, precision upgrades, and later lower-precision calls.
- Ensemble simulations share setup and coefficient statistics across beta
  values. Their assertions include deterministic coefficients with zero
  estimated variance, and fixtures restore the caller's NumPy RNG state.

When consolidating further, compare line and branch coverage for both compiled
and fallback runs. Check representative deliberately broken implementations as
well: coverage alone does not establish that assertions detect wrong answers.

Run the fixed adversarial audit with:

```sh
OPENBLAS_NUM_THREADS=1 python scripts/audit_adversarial.py --both-backends --output /tmp/fault-audit.json
```

It first runs an unmodified control and then injects each fault in a separate
process, without editing source files. The four representative faults cover
cumulant normalization, premature CRT reconstruction, stale HVP direction caches
and deterministic Haar sampling. A detected fault must produce an assertion
failure in its selected contract; import/collection failures and timeouts do not
count. This is a bounded regression audit, not an exhaustive mutation score.
