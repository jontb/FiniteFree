# P00 baseline and scope decision

**State: ready for review.** Recorded 2026-10-21, America/Toronto
(checks completed 2026-10-22 UTC). The contributor baseline passed. The selected
next feature is [P01: exact root moments and empirical matrix moments](p01-scope.md).
Its proposed signatures remain subject to the repository's API discussion rule.
No runtime implementation is part of this handoff.

## Source and evidence boundary

- Source: [`develop` at `2e13bc2832c5643df9c9d2730f5721f3b205121d`](https://github.com/jontb/FiniteFree/tree/2e13bc2832c5643df9c9d2730f5721f3b205121d).
  The live branch and local HEAD matched during the audit.
- Working branch: `docs/long-term-roadmap`, with the previously prepared README,
  generated index, navigation, roadmap and packet cards still uncommitted.
  Those changes were preserved. This report describes that working tree;
  upstream CI does not validate its uncommitted documentation.
- The inventory covers 102 `.py`/`.pyx` files in `finitefree`, `tests`, `scripts`
  and `examples`, excluding bytecode. Their before/after hashes agree. Its
  aggregate SHA-256 is
  `b5a1b4a906e56bc475053015febb8bbe3d5e832de60e182ae66b80fb5cdecd58`.
  The aggregate hashes the sorted path-to-SHA-256 map encoded as JSON with
  `sort_keys=True, separators=(',', ':')`; it is distinct from the numerical
  benchmark's inventory.
- Local receipts: `experiments/packets/P00/run-001/`. They include source and
  environment inventories, live GitHub responses, commands, complete logs,
  exact scope-oracle calculations and an evidence manifest. This directory is
  ignored; the result and receipt digests are retained below for review.

## Repository and backlog reconciliation

The exact source commit had [15/15 successful observed CI checks](https://github.com/jontb/FiniteFree/actions/runs/39699024386):
the 12 Linux/macOS/Windows and Python 3.10–3.13 combinations, minimum-runtime
dependencies, Chromium interaction checks and candidate artifacts. No PR was
open in the live snapshot. This records observed checks, without inferring a
fresh release approval or repository protection policy.

| Item | Evidence at the pinned source | Disposition |
| --- | --- | --- |
| [#35: PyPI quick example](https://github.com/jontb/FiniteFree/issues/35) | Merged [#39](https://github.com/jontb/FiniteFree/pull/39); `PYPI_README.md`, example checker and its regression are present. Issue remains open and assigned to PRINCE2-AI. | Implemented; avoid duplicate work. Issue-state reconciliation is a separate action. |
| [#36: seeded streaming example](https://github.com/jontb/FiniteFree/issues/36) | Merged [#40](https://github.com/jontb/FiniteFree/pull/40); `docs/gue-workflow.md` contains the seeded retained/streaming comparison and direct eigenvalue-count reference. Issue remains open. | Implemented; executable example passed in this baseline. |
| [#37: finite-state DPP tutorial](https://github.com/jontb/FiniteFree/issues/37) | No assignee, comments or open PR in the snapshot. Existing `tests/test_dpp.py` has an independent three-state probability oracle. | Available for separate scoping after another ownership check; not started by P00. |
| [#34: numerical preparation and cache](https://github.com/jontb/FiniteFree/pull/34) | Merged; lazy successful count-law caching and independent returned objects are present in `numerical.py` and its regression. | Existing capability, not a new optimization proposal. |
| Earlier CRT and callable-DPP defects | The [0.3.0 changelog](https://github.com/jontb/FiniteFree/blob/2e13bc2832c5643df9c9d2730f5721f3b205121d/CHANGELOG.md#030) and regressions retain the proved CRT bound and rejection of complex callable values before conversion/RNG use. | Fixed in current source; both suites and injected-fault controls passed. |

No issues, comments, PRs or repository settings were changed during this audit.

## Environment and contributor validation

A fresh `/tmp/finitefree-p00-env` used Python 3.13.5, GCC 14.2.0 and Linux
6.12.113 / glibc 2.41 on an AMD Ryzen 5 5500 with 12 visible CPUs. Runtime
dependencies were NumPy 2.5.4, SciPy 1.18.1, SymPy 1.14.0 and python-flint 0.9.0.
Tools included Ruff 0.17.1, mypy 2.4.0, pytest 9.1.1, Cython 3.3.0,
setuptools 76.1.0, MkDocs 1.6.1, Material 9.7.7 and mkdocstrings 1.0.8.
`dependencies.txt` contains the complete resolved environment; `uv pip check`
reported all 55 installed packages compatible.

Preparation and baseline commands, from the repository root:

```bash
UV_CACHE_DIR=/tmp/finitefree-p00-uv-cache uv venv --python /usr/bin/python3 /tmp/finitefree-p00-env
UV_CACHE_DIR=/tmp/finitefree-p00-uv-cache uv pip install --python /tmp/finitefree-p00-env/bin/python '.[dev]' Cython 'setuptools<77' mkdocs-material 'mkdocstrings[python]' build
UV_CACHE_DIR=/tmp/finitefree-p00-uv-cache uv pip check --python /tmp/finitefree-p00-env/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 /tmp/finitefree-p00-env/bin/python scripts/check.py --full
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /tmp/finitefree-p00-env/bin/python experiments/packets/P00/run-001/oracle-check.py
```

The first dependency download failed because sandbox DNS was unavailable; the
authorized network retry succeeded. `setup.log` and `setup-network.log` preserve
both outcomes. This was setup failure, not a failing library check.

| Check | Result | Receipt |
| --- | --- | --- |
| Ruff lint and formatting | Passed; 102 files already formatted | `baseline-full.log` |
| Strict mypy | Passed; 101 source files | `baseline-full.log` |
| Source Cython extension | Built successfully in place | `baseline-full.log` |
| Compiled source/tooling suite | 406 passed, 1 skipped; 407 collected | `baseline-full.log` |
| Flag-fallback source/tooling suite | 406 passed, 1 skipped; 407 collected | `baseline-full.log` |
| Adversarial audit | All 16 faults detected after clean controls | `baseline-full.log` |
| README/index synchronization | Passed | `baseline-full.log` |
| Executable documentation | All 54 Python examples passed | `baseline-full.log` |
| Showcase and interactive generators | Passed; two geometry and five spectral HTML assets generated | `baseline-full.log` |
| Strict MkDocs | Passed | `baseline-full.log` |
| P01 fixed-value scope checks | Passed using rational root powers, exact matrix-power traces and fixed finite-R values | `oracle-check.log` |
| Final documentation and whitespace | Passed after adding the handoff; 54 examples and strict build | `documentation-final.log` |

The final documentation commands used the same interpreter and thread/plugin
overrides above, with `PYTHONPATH` set to the repository root:

```bash
/tmp/finitefree-p00-env/bin/python scripts/sync_docs.py
/tmp/finitefree-p00-env/bin/python scripts/check_readme_examples.py
/tmp/finitefree-p00-env/bin/python -m mkdocs build --strict
git diff --check
```

The one skip in each suite is `test_real_gpu_parity`: CuPy is absent.
`nvidia-smi` could not communicate with the driver; this does not establish
whether a physical GPU is installed. GPU execution, GPU performance and the
optional-CuPy environment are unvalidated. The fallback flag selects documented
grid/CRT paths; it is not an extension-free installation. The source suite also
contains subprocess checks that block the native import.

The saved earlier `historical-cupy-typing.log` reports mypy `unused-ignore` and
`import-untyped` at `core.py:779` with an installed untyped CuPy. That import and
its `import-not-found` ignore remain in current source. This P00 environment has
no CuPy and passes mypy, so the environment-specific failure is **not freshly
reproduced or resolved**. Recreate its dependency configuration before changing
the ignore or classifying it as an active portability defect. It does not affect
the selected CPU-only P01 implementation path.

Installed-wheel tests, a new wheel/sdist audit, the full platform/dependency
matrix, browser automation and the full `numerical-v1` CPU/GPU measurements were
not rerun locally. The upstream checks above and historical receipts below have
their own source boundaries; they do not replace those candidate-level checks.

## Existing capabilities and conventions

The [code map](code-map.md), [API contracts](api.md) and
[test contracts](https://github.com/jontb/FiniteFree/blob/2e13bc2832c5643df9c9d2730f5721f3b205121d/tests/README.md)
were checked against the source. Public imports are collected in
[`finitefree/__init__.py`](https://github.com/jontb/FiniteFree/blob/2e13bc2832c5643df9c9d2730f5721f3b205121d/finitefree/__init__.py).

| Area | Present contract relevant to the next work |
| --- | --- |
| Exact algebra | Rational coefficient normalization, nonmonic scalar invariance, coefficient convolutions and lazy root/domain validation. Finite-R supports larger ambient dimension by zero-root embedding. There is no public normalized root-moment operation or inverse finite-R. |
| Precision and ownership | Numerical roots/outputs have explicit range and precision limits. Exact coefficients and certified root statements differ from floats. Owned inputs, read-only coefficient/root exports and restored precision contexts must remain intact. |
| GOE/GUE/GSE | Dense normalized Hermitian samples. GOE diagonal variance is `2/d`, off-diagonal variance `1/d`; GUE diagonal and squared off-diagonal magnitude have mean `1/d`. GSE uses a physical `2d` representation with duplicated eigenvalues. `scale` multiplies the matrix. |
| Wishart and Haar | Wishart beta 1/2 uses `XX*/n`; beta 4 uses a physical `2d` representation normalized by `2n`, with one eigenvalue per pair in degree-d comparisons. Haar orthogonal/unitary/symplectic samplers exist. General beta compact samplers and Jacobi/MANOVA remain future work. |
| RNG | Omitted/None RNG preserves global NumPy legacy draw order. An explicit `Generator` is consumed in place. Integer seeds and `RandomState` are not accepted RNG objects. New helpers should consume no randomness themselves. |
| Expected polynomials | Exact scaled Hermite/Laguerre expectation polynomials are available. Their root moments generally differ from expected sampled spectral moments. |
| Empirical comparisons | Retained comparisons store eigenvalues and characteristic coefficients. Streaming comparisons retain scaled coefficient moments and closed-interval count histograms, with bounded storage. They do not retain per-sample nonlinear transform summaries. |
| Prepared numerics | Prepared pencils, Hermite-aware kernel evaluation, quadrature and count laws already exist. Lazy successful count-law caching returns independent results and retries failed validation. These are numerical estimates under their documented contracts. |

The current private `_sample_observations` helper checks finite Hermitian input,
physical shape and paired spectra, then also forms characteristic coefficients.
That last step can overflow even if low-order moments are representable. P01
must not require characteristic-polynomial construction just to obtain moments.
No changes to the retained/streaming comparison contracts are needed for P01.

## Empirical transform inventory and assigned gaps

All five current public transform entry points accept polynomial objects. None
provides a supported direct matrix/spectrum route. A caller's hand-built
polynomial is not a complete empirical interface: input validation, numerical
limits, pairing and aggregation still need a documented contract.
The [required counterpart rule](implementation-packets.md#required-empirical-matrix-counterparts)
therefore remains an implementation gap for existing transforms.

| Operation and current polynomial path | Required empirical target and independent comparison | Assigned resolution |
| --- | --- | --- |
| Cauchy: `p'(z)/(degree*p(z))`, symbolic, positive degree | Per-matrix normalized resolvent trace, away from poles; compare diagonal spectra and direct small-matrix inversion. State physical/effective dimension and zero embedding. | P01 establishes the small matrix-input convention; [P16](packets/p16-empirical-legacy-transforms.md) supplies the supported Cauchy route. |
| Finite R: normalized elementary coefficients, classical cumulant recurrence and finite-d scaling; exact or centered Arb numerical path; orders above ambient d are zero | Per-matrix finite cumulants, or explicitly labeled cumulants of estimated expected coefficients. Compare independent determinant coefficients and partition/formal-series values. A limiting R-transform of the mean spectral measure is a different operation. | [P02](packets/p02-inverse-finite-r.md), including the forward empirical path and corresponding inverse inputs. |
| S: normalized adjacent coefficient ratios at `-k/d`; ordinary roots strictly positive; exact rationals or float64 | Per-matrix discrete finite S on positive-definite samples, with explicit coefficient/range failures; exact positive diagonal reference and a full-rank Wishart example. Current float output has no final-finiteness guarantee. | P16. Unitary objects bypass positivity but still have rational/nonzero coefficient restrictions; general complex-unitary coefficient support is a separate scope. |
| T: nonnegative roots, zero multiplicity and right-continuous steps on `0<t<1`; exact ratios | Per-matrix finite T, preserving dimension, zero atoms and boundary convention; compare exact PSD diagonal samples and a rank-deficient model with declared rank. No silent zero threshold or root repair. | P16. |
| Symmetric S: even degree, even polynomial, ratio default; `standard` adds real-root validation and the positive-imaginary square-root branch | Per-sample transform of a genuinely symmetric spectrum; compare exact `[-3,-1,1,3]` and a Hermitian block linearization with eigenvalues in ± pairs. Preserve zero multiplicity, nodes and convention. | P16. Generic GOE/GUE draws are not exactly ± symmetric; duplicated quaternionic eigenvalues are a separate pairing condition. |

P16 is a required legacy-support packet, added from this inventory. It is
separate from optional C01, which studies new transform definitions. P01 remains
bounded to moments; implementing P01 alone does not close the other rows.

Three statistical targets must be labeled separately: the mean of per-sample
transforms, a transform of the estimated expected characteristic polynomial,
and a transform of an estimated mean spectral measure. Nonlinear operations
generally do not commute with averaging. An averaged polynomial can also fail
the required root domain; no implicit positivity or symmetry repair is allowed.
For fixed effective dimension, moments themselves are linear in the spectral
measure. Across independent matrices, report uncertainty at the matrix level.
Existing coefficient means cannot recover unrecorded per-sample nonlinear
statistics.

## Prior numerical receipts and remaining evidence

These were inspected and hashed, not rerun as P00 measurements.

| Local receipt | Supported conclusion | Remaining limit |
| --- | --- | --- |
| `experiments/numerical-prep-2026-10-20/validation.json` | Benchmark tooling and five-case smoke were validated at its recorded source inventory. | Full baseline explicitly unrun. |
| `experiments/test-consolidation-2026-10-20/validation.json` | Historical compiled/fallback executed Python lines/arcs were preserved while collection fell from 440 to 404. | No proportional speedup or native branch-coverage claim. Current suite now collects 407. |
| `experiments/quadrature-cache-2026-10-20/validation.json` and its plan/raw JSON | Six repeated prepared queries changed from six eigendecompositions to one. Three alternating process pairs at ranks 2/32 had identical outputs and improved warm queries. | First-query latency did not improve; no whole-workflow or GPU claim. |
| `experiments/pr34/committed-smoke/manifest.json` | `numerical-v1` smoke completed 5/5 cases at clean commit `2a1e9ad21c3b46529e7357ed29c0cbff8137edd5`, seed 862 and one-thread child processes. | CPU pencil/kernel/summary smoke only. Full CPU baseline and CuPy remain unrun. |

The committed-smoke source/tool hash is
`eced6f03f1c2be4399026e4c3ef73f39fe2384bc6c3e76c85e8d8b5784a51597`.
The complete prior-receipt paths and file hashes are in `prior-receipts.json`.
The full CPU measurement remains unrun under the existing
[numerical-v1 protocol](benchmark-plan.md). P00 launches no benchmark sweep or
streaming merge/checkpoint implementation.

## First feature and handoff

Proceed next with the bounded [P01 scope proposal](p01-scope.md): exact normalized
root moments plus empirical moments of each actual Hermitian sample, with
explicit ambient dimension and paired representations. Independent values were
checked for roots `[1,3]`, zero embedding, a complex Hermitian matrix and the
two-sample aggregation counterexample. No proposed API has been implemented.

The next implementation prerequisite is to record the public API/convention
agreement in an issue, as required by
[CONTRIBUTING.md](https://github.com/jontb/FiniteFree/blob/2e13bc2832c5643df9c9d2730f5721f3b205121d/CONTRIBUTING.md).
This local proposal is reviewable, but does not claim that such a discussion or
maintainer acceptance already occurred. The existing baseline has no unexplained
failure on the chosen CPU path. Optional-backend and candidate-level limits are
recorded above.

This handoff was prepared with material agent assistance: source and issue/CI
inspection, isolated-environment checks, exact scope calculations and drafting.
Changes are documentation and ignored evidence only. Merge, publication and
runtime changes are outside P00.

## Receipt digests

All paths below are relative to `experiments/packets/P00/run-001/`.
The evidence manifest inventories individual receipt hashes; these fixed
digests make the local evidence identifiable even when it is not in Git.

| Receipt | SHA-256 |
| --- | --- |
| `baseline-full.log` | bd40f9543d9db3b35b794492f786a6d88e4f0e4b16bbb77404ef769603cd1f8e |
| `github-live.json` | a8df57a23d1f2db8d4d7a1bb36b8ab757995fbf0546244ccbf1ed1d105ecfedc |
| `environment.json` | 9c1089f943f51f45d6c00aef3286a1cdcbd6b98ec6c6c5de1276e62953616b96 |
| `oracle-check.log` | af3dc09d203f4d975a4101fbe39ed3fb3b55df79093d4f15bf0b7d845c6be5b8 |
| `prior-receipts.json` | 4d280d441209fbff0c05662d7deefb825846b5a08de45ffbfb3ad86288def394 |
| `evidence-manifest.json` | 5ca47dbc140f7e8a0d2594f0cab5961c376520850e046e6870410912c99997ef |
