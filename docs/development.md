# Development and documentation

These instructions support development of FiniteFree. Production versions are listed on [PyPI](https://pypi.org/project/finitefree/); `main` and the public docs track the latest release, while `develop` accepts ongoing work. Follow the [release process](release.md) for candidate validation and publication, and record the Git commit when reproducing results.

## Branch workflow

The [contribution guide](https://github.com/jontb/FiniteFree/blob/develop/CONTRIBUTING.md)
describes issue/PR expectations and the zero-required-approvals maintainer policy.
The [code map](code-map.md) identifies implementation and validation boundaries.

`main` contains the functional API of the latest published PyPI release. Its README and the public MkDocs site describe that release. Documentation and workflow maintenance may reach `main` between releases when it preserves that functional API and accurately describes the published package.

Start feature and maintenance branches from `develop`, and open their pull requests against `develop`. CI runs on these pull requests and on pushes to `develop`. Keep API changes, their tests and documentation together there; unreleased features must not enter `main` or its published documentation.

GitHub Pages deploys only from `main`. Pushes to `develop` do not deploy it, and a manual documentation run from any other branch is skipped. CI still checks examples and builds documentation for development changes.

For a release, validate the exact candidate on `develop`, publish its approved version tag, then verify the production package before promoting that released source through a `develop` → `main` pull request. Follow the [release process](release.md#release-promotion). Bring any merge commit or stable maintenance from `main` back into `develop` before starting the next release.

## Installation

Use the [README setup instructions](index.md#installation) to create a virtual environment with Python 3.10 or later. CI covers CPython 3.10–3.13 on Linux, macOS and Windows, plus the exact runtime floors in `requirements/minimum-runtime.txt` on Python 3.10. Verify the matrix on the intended release commit. The former Python 3.9 / FLINT 0.6 declaration failed an actual rational multivariate construction; see [release compatibility](release.md#compatibility).

For standard installation, tests and documentation:

```bash
python -m pip install ".[dev]"
python -m pip install Cython "setuptools<77" build mkdocs-material "mkdocstrings[python]"
```

The Hatch backend builds the Cython extension in the wheel. To make it available when tests explicitly import this source checkout, build it in place as CI does:

```bash
python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy as np; setup(ext_modules=cythonize(Extension('finitefree.utils.modular_fast', ['finitefree/utils/modular_fast.pyx'], include_dirs=[np.get_include()]), force=True), script_args=['build_ext', '--inplace'])"
```

Source builds require a C compiler. Building python-flint itself additionally requires FLINT/GMP/MPFR headers and libraries; compatible python-flint wheels avoid that build. Editable installation is useful during development, but installed-wheel verification below requires a **standard** installation without the checkout on the import path.

## Checks

After installing the development dependencies, Cython and `setuptools<77`, run
`python scripts/check.py` for lint, formatting, typing, an in-place extension
build and source/tooling tests. Add `--full` for the fallback suite, examples,
README/index consistency, adversarial fault audit and strict MkDocs build (install
its dependencies below). The full runner generates both required interactive
asset families before building documentation.
Add `--installed-wheel` after a standard installation of the current candidate
to validate it away from the source tree. The runner uses the active interpreter
and stops at the first failure; the individual commands remain available below.

Run these commands from the repository root in Bash (including Git Bash on Windows):

```bash
python -m ruff check finitefree tests scripts examples
python -m ruff format --check finitefree tests scripts examples
python -m mypy finitefree tests scripts examples
PYTHONPATH=. python -m pytest --import-mode=importlib tests/ scripts/tests/
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/ scripts/tests/
PYTHONPATH=. python scripts/check_readme_examples.py
PYTHONPATH=. python examples/showcase.py
python scripts/test_installed_wheel.py
python -m build
```

To reproduce the minimum-runtime job in a fresh Python 3.10 environment, install the pinned runtime before the package and build against its NumPy headers:

```bash
python -m pip install --only-binary=python-flint -r requirements/minimum-runtime.txt
python -m pip install "pytest==7.0.0" hatchling hatch-cython Cython "setuptools<77" wheel build
python -m pip install --no-deps --no-build-isolation .
python -m pip check
```

Then build the source extension and run both source suites, examples and the installed-wheel helper above. Building with isolation would select newer build dependencies and would not check the minimum NumPy headers.

The installed-wheel helper runs compiled and flag-fallback library suites explicitly
in a temporary directory and confirms that the compiled extension is present.
Documentation-tool tests require the source tree. `PYFFP_DISABLE_CYTHON=1` selects
Python grid evaluation and CRT reconstruction; it does not remove the built
extension, and `modular_det` can still use it. A focused fresh-process test blocks
the extension import separately and checks determinant, CRT and specialization
results against independent values. It does not recursively launch pytest or
claim an extension-free distribution. In PowerShell, set the corresponding
variables with `$env:PYTHONPATH="."` and `$env:PYFFP_DISABLE_CYTHON="1"`, then clear
them before installed-wheel checks.

## Documentation

README is the source for `docs/index.md`. `PYPI_README.md` is the concise package-index description selected by `pyproject.toml`; keep its API claims aligned with these docs and use absolute links. Edit README, API prose, tutorial and docstrings; regenerate the index:

```bash
PYTHONPATH=. python examples/interactive_geometry.py --output docs/visuals/interactive
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python examples/spectral_dashboards.py --output docs/visuals/interactive
python scripts/sync_docs.py
PYTHONPATH=. python scripts/check_readme_examples.py
python -m mkdocs build --strict
```

The two generation commands must run before the strict build: they write all seven self-contained HTML companions under `docs/visuals/interactive/`, which MkDocs copies to `site/visuals/interactive/`. Generated files are ignored by Git. CI and the guarded documentation-deployment build run the same commands. Open the built home page to use the links beside each visual, or run `python -m mkdocs serve` after generation. GitHub README links point to the sections below; the synchronizer rewrites those links to the generated HTML in the built home page. `OPENBLAS_NUM_THREADS=1` keeps the spectral generation's matrix work single-threaded.

The synchronizer accepts single-level `<details>` blocks with separate-line opening, summary and closing tags. It preserves fenced examples and aborts before writing on malformed details or missing input. The example checker runs Python fences independently in README, the PyPI description (`PYPI_README.md`), the generated index, API, tutorial, development guide and changelog. Examples must import their own dependencies and contain meaningful assertions where practical.

The strict build treats missing documentation targets and invalid anchors as warnings that fail the build. README's relative image paths also point to repository assets; generated-site assets live under `docs/visuals/assets`. External reference accessibility is a separate check and can be limited by publisher access restrictions.

Check that regeneration produces the expected index, then commit both README and the index. User-facing README/API text explains current behavior; dated measurements and migration history belong in changelog or review notes.

## Interactive geometry examples

Two standalone explorers are included in the 0.4.0 source and linked from the public site. Run their generators from a source checkout; the installed library wheel does not include the example scripts.

```bash
PYTHONPATH=. python examples/interactive_geometry.py
```

Open `visuals/generated/hyperbolicity-cone.html` and `visuals/generated/moving-line-roots.html` directly in a modern browser. Each file embeds its own styling, scripts and exact rational model data; opening it requires no server, network access or JavaScript runtime dependency. Use `--output PATH` to choose another output directory. Generated files are ignored by Git; editable templates live in `visuals/interactive/`.

### Hyperbolicity cone

[Open the hyperbolicity cone explorer](visuals/interactive/hyperbolicity-cone.html) in a built documentation site. For a source checkout, generate and open `visuals/generated/hyperbolicity-cone.html` as described above.

The cone explorer constructs the symmetric 3 × 3 determinant through `from_symmetric_matrix_pencil`, substitutes `t = 1`, and exports exact sparse coefficients. Its rotating surface and linked slice separate the PSD section from other determinant-sign chambers. Sliders and boundary/chamber presets expose eigenvalues and principal minors, including the degenerate `|z| = 1` slice. The browser classifies numerical membership with a tolerance; determinant sign alone does not establish PSD.

### Moving line roots

[Open moving line roots](visuals/interactive/moving-line-roots.html) in a built documentation site. For a source checkout, generate and open `visuals/generated/moving-line-roots.html` as described above.

The line explorer uses `det(xI + yD + zB)` with `D = diag(-3,-1,1,3)` and path adjacency `B`. It generates 3,321 exact rational control-grid restrictions in the identity direction, together with univariate derivatives checked against restricted multivariate partial derivatives. Root tracks, the selected polynomial, derivative interlacing and exact coefficient readouts update together. Zero coupling permits crossings and repeated roots; nonzero coupling opens gaps. Colors follow diagonal identities at zero coupling and ascending root ranks otherwise. Numerical rendering uses Float64, not a general hyperbolicity or stability certificate.

The Python example/data checks run with the source-only tooling suite:

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib scripts/tests/test_interactive_geometry.py
```

Optional browser checks use a development-only Playwright driver and an installed Chromium/Chrome. They exercise sliders, reset, linked clicks, keyboard controls, animation, near-boundary and zero-coupling cases, and responsive layouts, saving labeled screenshots and a validation receipt:

```bash
npm install --prefix /tmp/finitefree-browser playwright-core@1.63.0
NODE_PATH=/tmp/finitefree-browser/node_modules node visuals/interactive/browser-check.cjs visuals/generated visuals/generated/screenshots
```

Set `FINITEFREE_BROWSER` to the browser executable if it is not on PATH. CI runs this check on Ubuntu. Neither Playwright nor Node is needed to open the generated HTML. Existing batch visual scripts and their optional Matplotlib/Pillow dependencies are unaffected.

For managed browsers that disallow `file://`, serve only the generated directory with `python -m http.server 8765 --bind 127.0.0.1 --directory visuals/generated`, then set `FINITEFREE_BASE_URL=http://127.0.0.1:8765/` on either browser-check command. The optional transport accepts only loopback HTTP; Chromium sandboxing remains enabled and any non-navigation network request still fails the checks. The standalone files themselves need no server in browsers that permit local HTML.

## Spectral dashboards

For an end-to-end example connecting existing samplers, expected polynomials,
DPP mean density and empirical comparisons, see the [GUE spectral workflow](gue-workflow.md).

Five dashboards consolidate the existing spectral visualization families into shared controls and paired plots:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python examples/spectral_dashboards.py
```

Open the generated HTML files directly; each embeds its data, styling and scripts and makes no external requests.

### Ensemble convergence

[Open ensemble convergence](visuals/interactive/ensemble-convergence.html) in a built documentation site, or generate and open `visuals/generated/ensemble-convergence.html` from the source checkout.

Choose real/Hermitian Wigner, GSE or real/complex/quaternionic Wishart; applicable Gaussian/Rademacher/uniform coordinates; or deterministic Hermite/Laguerre/Legendre/compound-Wishart roots. Vary size, aspect ratio and recorded sample.

The compound-Wishart option keeps τ = 1: degree d, d multiplicative factors, and n = d². It records only d = 8,16,32,64, using exact normalized coefficients of the mean-one Wishart polynomial raised to the d-th power and scoped 192-bit root isolation. There is no expensive root solve in the browser. Tests independently construct the coefficients and compare all recorded roots with a 384-bit Arb reference; displayed Float64 values are not interval certificates.

The mean-one free-lognormal convention is S(w) = exp(−w), with moments m₀,m₁,m₂,m₃ = 1,1,2,11/2. The finite coefficient ratio is (1−(k−1)/d²)^(−d), tending to exp(t) when k/d → t, consistent with S(−t). The boundary relation x = (1+w)exp(w)/w is solved with w = a−ib: bracket b cot(b)−b² = a(a+1) on 0 < b < π/2, as a runs between (−1±√5)/2. This follows the positive-density branch across the complete support [0.0757393348,4.8571781362] without unchecked complex Newton iteration. The density is b/(πx). A 2,049-point endpoint-clustered grid is integrated and normalized by its raw mass (shown in the dashboard); tests check raw moments, the off-support Stieltjes equation and refinement to 4,097 points. This is a numerical reference, not a certified density enclosure.

### Finite transforms

[Open finite transforms](visuals/interactive/finite-transforms.html) in a built documentation site, or generate and open `visuals/generated/finite-transforms.html` from the source checkout.

Switch between the Wishart finite T-transform and additive-CLT cumulants; vary dimension, aspect ratio and probe t.

### Convolution interlacing

[Open convolution interlacing](visuals/interactive/convolution-interlacing.html) in a built documentation site, or generate and open `visuals/generated/convolution-interlacing.html` from the source checkout.

Switch between additive and multiplicative convolution and vary the exact rational input-root shift.

### Hermite kernels

[Open hermite kernels](visuals/interactive/hermite-kernels.html) in a built documentation site, or generate and open `visuals/generated/hermite-kernels.html` from the source checkout.

Switch between bulk/sine and soft-edge/Airy kernel sections or continuous gap / edge CDF. Vary the projection rank; in the CDF view, the threshold slider selects s in [−4,3].

For the weight e^(−x²), a_d = √(2d) and s_d = 1/(√2 d^(1/6)). The finite-rank curve approximates P((λ_max−a_d)/s_d ≤ s) using the Hermite continuous gap determinant, and the separate Tracy–Widom β = 2 reference uses the Airy kernel. Both use Gauss–Legendre Nyström determinants on the rescaled interval [s,10] with 64 nodes. Precomputation checks 96 nodes and cutoff 14 at every one of the 71 thresholds for all six ranks 8–256 and the Airy reference, rejecting discrepancies above 10⁻⁹. Controls only select stored curves and thresholds. The displayed sensitivity is empirical, not a certified quadrature or tail bound.

Independent tests cover the rank-one Gaussian CDF, a finite-rank half-line Gram determinant integrated with explicit Hermite polynomials, and the Airy integral-kernel identity. Numerical tail-trace integrals assess the omitted half-line separately. The finite-rank curve is not labeled Tracy–Widom. See [Bornemann's numerical RMT review](https://arxiv.org/abs/0904.1581) for the Fredholm/Nyström formulation and edge scaling. The batch HKPV figure remains a distinct approximate sampling illustration; it does not supply this dashboard's reference.

### Unitary root flow

[Open unitary root flow](visuals/interactive/unitary-root-flow.html) in a built documentation site, or generate and open `visuals/generated/unitary-root-flow.html` from the source checkout.

Vary degree and time in the polynomial phase portrait and angular root tracks.

### Interpretation and validation

Ensemble plots distinguish a single sampled ESD from an expected-polynomial root measure. Covariance normalization is `XX*/n`, with γ = d/n and zero atom `max(0,1−1/γ)`; the quaternionic complex representation uses `XX*/(2n)` and one eigenvalue per pair. Histograms exclude the structural atom and retain probability normalization; CDFs retain all finite mass. Recorded seeds are reproducible within the numerical environment, with independently generated matrices across dimensions. Bounded-entry models use independent centered, variance-one coordinates and Wigner off-diagonal variance 1/d; GSE remains Gaussian. Gaussian samplers and polynomial construction use public FiniteFree APIs. Rendering and matrix diagonalization are Float64 approximations, not certificates.

Wishart T-transform steps retain exact rational values and the right-continuous jump convention. The additive CLT uses even square dimensions so its dilation is rational; finite variance is d/(d−1), rather than exactly one. Hermite kernel sections use normalized function recurrence, not random ESDs. Unitary roots at t = 0 are inserted from the exact coalesced polynomial; positive-time roots retain the public companion solver's numerical radial defect.

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib scripts/tests/test_spectral_dashboards.py
NODE_PATH=/tmp/finitefree-browser/node_modules node visuals/interactive/dashboard-check.cjs visuals/generated visuals/generated/dashboard-screenshots
```

CI generates both the geometry explorers and spectral dashboards, exercises dropdowns, slider endpoints, jumps, zero atoms, sample selection, sweep/reset and desktop/tablet/phone layouts, and saves the HTML and screenshots as a development artifact. Mathematical references include [Menon's random matrix notes](https://www.dam.brown.edu/people/menon/publications/rmt-2021.pdf), [S-transform in finite free probability](https://arxiv.org/abs/2408.09337), and [unitary Hermite polynomials](https://arxiv.org/abs/2203.05533). The compound-Wishart and continuous-gap families are consolidated into the ensemble and Hermite dropdowns. The older batch scripts are not imported by these generators.

## Benchmarks

For the prepared-pencil, prepared-kernel and streaming measurement plan, fixed
inputs, bounded runner and optimization gates, see the
[numerical benchmark protocol](benchmark-plan.md). Its smoke profile validates
tooling; larger measurements are a separate run.

These small cases exercise the root, transform, scalar-kernel and projection
tools. They write JSON only at the supplied output paths:

```bash
PYTHONPATH=. python scripts/benchmark_roots.py --degrees 32 --output roots-benchmark.json
PYTHONPATH=. python scripts/benchmark_compound_roots.py --degrees 10 15 30 32 --precision 192 --repeats 3 --output compound-benchmark.json
PYTHONPATH=. python scripts/benchmark_transforms.py --degrees 60 --output transform-benchmark.json
PYTHONPATH=. python scripts/benchmark_kernels.py --degrees 20 --output kernel-benchmark.json
PYTHONPATH=. python scripts/benchmark_projection.py --degrees 100 --dimension 20 --output projection-benchmark.json
```

Larger defaults are optional and can be expensive. Root tools use independent Arb references; compound calls include domain validation and report cached latency separately. Transform tools compare shifted Hermite/Wishart analytic cumulants with cached normalized coefficients. Kernels use exact rational basis sums with warmed evaluation caches. Projections compare exact family derivative formulas, excluding construction/reference generation from timing. No benchmark establishes a universal accuracy or speed bound.

For scaling studies, record the Git commit, script digest, dependency versions, input families and coefficient bit sizes. Run bounded serial subprocess repetitions with BLAS thread settings established before imports, and preserve individual timings and failures. Distinguish a first public call (including any lazy imports), a fresh-object call after warming the process, and a cached same-object call. Keep construction, reference generation and validation timing explicit; generic public root calls can include certification. Record native process peak RSS with its import/setup contribution, and verify independent accuracy references outside timing. Plot observed ranges and describe empirical slopes only for the measured inputs; they do not prove asymptotic complexity.

The [multivariate scaling study](multivariate.md#reproducible-scaling-comparison) compares all numerical and context APIs with an explicit local Git baseline. Its smoke suite verifies the workflow quickly; the standard suite records degree/support/bit-size and batch scaling, interleaved timing samples, traced allocations, profiles and representative fresh-process RSS. Set thread limits before imports and distinguish output storage from scratch space.

`visuals/*.py` regenerate pre-rendered illustrations and require Matplotlib/Pillow. They may write many frames and use high-degree solvers or approximate grid sampling; their output is illustrative rather than part of the regression suite.

## Release validation

The 0.4 series retains the 0.3.0 compatibility floors and symmetric S conventions; 0.4.1 changes metadata and documentation only. See the [migration notes](https://github.com/jontb/FiniteFree/blob/v0.4.1/CHANGELOG.md). For future releases, follow the [release review checklist](release.md) to inspect artifacts and checks on the final versioned commit, including consolidated tests. The publish workflow uses `v*.*.*` tags or a manual dispatch and trusted publishing; invoking either is a separate release action.

## Prepared pencil validation checklist

The initial cloud inspection on 2026-10-09 found no GPU and supported only CPU
verification. Subsequent genuine-device correctness validation on 2026-10-18
passed on an NVIDIA RTX 3050 (8 GiB), Python 3.12.15, CuPy 14.2.0 and CUDA runtime
13.2. The float64 tests cover determinant/gradient/HVP parity, empty batches,
residency, zero/rank-deficient matrices and mixed batches. The original prepared
suite passed all 18 tests with warnings treated as errors. Consolidated contracts
retain those cases; mocks are labeled separately and do not supply GPU evidence.

- [x] Prepared float64 coefficients, batch evaluation, reusable pivoted LU,
  slogdet, log-absolute-determinant gradient and HVP.
- [x] CPU analytic/exact/finite-difference coverage, singular and near-singular
  cases, zero coordinates, empty/strided arrays, ownership and backend absence.
- [x] Benchmark harness separates setup, transfers, synchronized execution,
  reuse, partial memory measurements, errors and failures.
- [x] Record genuine-device metadata and float64 correctness tests, including
  CPU parity and singular handling, on the local RTX 3050 configuration above.

The following performance and expansion milestones are deferred beyond the
0.3.0 correctness scope:
- [ ] Measure cold startup and warm latency, all transfer costs and total peak
  device/native memory; retain individual samples and failure records.
- [ ] Measure crossover batch sizes for matching matrix/variable sizes and
  accuracy. Do not publish speed claims before obtaining that evidence.
- [ ] Later milestones: compiled sparse-polynomial GPU evaluation and exact
  modular GPU construction (outside prepared-pencil scope).

### Internal spectral workflow track

Keep this separate from prepared-pencil implementation. Compose existing GUE
sampling, expected characteristic polynomials, projected orthogonal bases, DPP
kernels and empirical comparisons using one consistent Gaussian measure. Reuse
exact-verified scaled Hermite recurrence data where it avoids monomial evaluation;
preserve generic fallbacks and public signatures. Include a runnable workflow and
exact integration checks. The finite-size mean ESD differs from the zero measure
of the expected characteristic polynomial. No new mathematical APIs are implied.

- [x] Bounded concentration showcase: `examples/gue_concentration.py` compares
  normalized interval counts across GUE matrix sizes with numerical count laws
  and separately reported Monte Carlo uncertainty. The [GUE workflow](gue-workflow.md)
  records seeds, sample sizes, refinement checks and unresolved tail probabilities.
  Increasing sample count is not increasing concentration in matrix size;
  rigorous quadrature enclosures remain deferred research.
