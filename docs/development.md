# Development and documentation

These instructions describe the current `develop` source checkout, including unreleased development examples. For the published package, use the [v0.2.0 source](https://github.com/jontb/FiniteFree/tree/v0.2.0) and install its matching [PyPI release](https://pypi.org/project/finitefree/0.2.0/) with `python -m pip install finitefree==0.2.0`. A local build alone does not establish production publication. Record the Git commit when reproducing results.

## Branch workflow

`main` contains the functional API of the latest published PyPI release. Its README and the public MkDocs site describe that release. Documentation and workflow maintenance may reach `main` between releases when it preserves that functional API and accurately describes the published package.

Start feature and maintenance branches from `develop`, and open their pull requests against `develop`. CI runs on these pull requests and on pushes to `develop`. Keep API changes, their tests and documentation together there; unreleased features must not enter `main` or its published documentation.

GitHub Pages deploys only from `main`. Pushes to `develop` do not deploy it, and a manual documentation run from any other branch is skipped. CI still checks examples and builds documentation for development changes.

For a release, validate the exact candidate on `develop`, publish its approved version tag, then verify the production package before promoting that released source through a `develop` → `main` pull request. Follow the [release process](release.md#release-promotion). Bring any merge commit or stable maintenance from `main` back into `develop` before starting the next release.

## Installation

Use the [README setup instructions](index.md#installation) to create a virtual environment with Python 3.10 or later. CI covers CPython 3.10–3.13 on Linux, macOS and Windows, plus the exact runtime floors in `requirements/minimum-runtime.txt` on Python 3.10. Verify the matrix on the intended release commit. The former Python 3.9 / FLINT 0.6 declaration failed an actual rational multivariate construction; see [release compatibility](release.md#compatibility).

For standard installation, tests and documentation:

```bash
python -m pip install ".[dev]"
python -m pip install Cython build mkdocs-material "mkdocstrings[python]"
```

The Hatch backend builds the Cython extension in the wheel. To make it available when tests explicitly import this source checkout, build it in place as CI does:

```bash
python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy as np; setup(ext_modules=cythonize(Extension('finitefree.utils.modular_fast', ['finitefree/utils/modular_fast.pyx'], include_dirs=[np.get_include()])), script_args=['build_ext', '--inplace'])"
```

Source builds require a C compiler. Building python-flint itself additionally requires FLINT/GMP/MPFR headers and libraries; compatible python-flint wheels avoid that build. Editable installation is useful during development, but installed-wheel verification below requires a **standard** installation without the checkout on the import path.

## Checks

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

The installed-wheel helper runs library tests in a temporary directory and confirms that the compiled extension is present. Documentation-tool tests require the source tree. `PYFFP_DISABLE_CYTHON=1` selects Python grid evaluation and CRT reconstruction; it does not remove the built extension, and `modular_det` can still use it. This checks those fallback paths rather than an extension-free installation. In PowerShell, set the corresponding variables with `$env:PYTHONPATH="."` and `$env:PYFFP_DISABLE_CYTHON="1"`, then clear them before installed-wheel checks.

## Documentation

README is the source for `docs/index.md`. Edit README, API prose, tutorial and docstrings; regenerate the index:

```bash
python scripts/sync_docs.py
PYTHONPATH=. python scripts/check_readme_examples.py
python -m mkdocs build --strict
```

The synchronizer accepts single-level `<details>` blocks with separate-line opening, summary and closing tags. It preserves fenced examples and aborts before writing on malformed details or missing input. The example checker runs Python fences independently in README, the generated index, API, tutorial, development guide and changelog. Examples must import their own dependencies and contain meaningful assertions where practical.

The strict build treats missing documentation targets and invalid anchors as warnings that fail the build. README's relative image paths also point to repository assets; generated-site assets live under `docs/visuals/assets`. External reference accessibility is a separate check and can be limited by publisher access restrictions.

Check that regeneration produces the expected index, then commit both README and the index. User-facing README/API text explains current behavior; dated measurements and migration history belong in changelog or review notes.

## Interactive geometry examples on develop

Two standalone explorers are available in the development branch. They are development examples; the stable API and public site remain tied to the latest published release.

```bash
PYTHONPATH=. python examples/interactive_geometry.py
```

Open `visuals/generated/hyperbolicity-cone.html` and `visuals/generated/moving-line-roots.html` directly in a modern browser. Each file embeds its own styling, scripts and exact rational model data; opening it requires no server, network access or JavaScript runtime dependency. Use `--output PATH` to choose another output directory. Generated files are ignored by Git; editable templates live in `visuals/interactive/`.

The cone explorer constructs the symmetric 3 × 3 determinant through `from_symmetric_matrix_pencil`, substitutes `t = 1`, and exports exact sparse coefficients. Its rotating surface and linked slice separate the PSD section from other determinant-sign chambers. Sliders and boundary/chamber presets expose eigenvalues and principal minors, including the degenerate `|z| = 1` slice. The browser classifies numerical membership with a tolerance; determinant sign alone does not establish PSD.

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

## Spectral dashboards on develop

Five dashboards consolidate the existing spectral visualization families into shared controls and paired plots:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python examples/spectral_dashboards.py
```

Open the generated HTML files directly; each embeds its data, styling and scripts and makes no external requests.

| File in `visuals/generated/` | Dropdowns and sliders |
| --- | --- |
| `ensemble-convergence.html` | Real/Hermitian Wigner, GSE, real/complex/quaternionic Wishart; applicable Gaussian/Rademacher/uniform coordinates; deterministic Hermite/Laguerre/Legendre roots; size, aspect ratio and recorded sample |
| `finite-transforms.html` | Wishart finite T-transform or additive-CLT cumulants; dimension, aspect ratio and probe t |
| `convolution-interlacing.html` | Additive or multiplicative convolution; exact rational input-root shift |
| `hermite-kernels.html` | Bulk/sine or soft-edge/Airy kernel section; projection rank |
| `unitary-root-flow.html` | Degree and time; polynomial phase portrait and angular root tracks |

Ensemble plots distinguish a single sampled ESD from an expected-polynomial root measure. Covariance normalization is `XX*/n`, with γ = d/n and zero atom `max(0,1−1/γ)`; the quaternionic complex representation uses `XX*/(2n)` and one eigenvalue per pair. Histograms exclude the structural atom and retain probability normalization; CDFs retain all finite mass. Recorded seeds are reproducible within the numerical environment, with independently generated matrices across dimensions. Bounded-entry models use independent centered, variance-one coordinates and Wigner off-diagonal variance 1/d; GSE remains Gaussian. Gaussian samplers and polynomial construction use public FiniteFree APIs. Rendering and matrix diagonalization are Float64 approximations, not certificates.

Wishart T-transform steps retain exact rational values and the right-continuous jump convention. The additive CLT uses even square dimensions so its dilation is rational; finite variance is d/(d−1), rather than exactly one. Hermite kernel sections use normalized function recurrence, not random ESDs. Unitary roots at t = 0 are inserted from the exact coalesced polynomial; positive-time roots retain the public companion solver's numerical radial defect.

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib scripts/tests/test_spectral_dashboards.py
NODE_PATH=/tmp/finitefree-browser/node_modules node visuals/interactive/dashboard-check.cjs visuals/generated visuals/generated/dashboard-screenshots
```

CI generates both the geometry explorers and spectral dashboards, exercises dropdowns, slider endpoints, jumps, zero atoms, sample selection, sweep/reset and desktop/tablet/phone layouts, and saves the HTML and screenshots as a development artifact. Mathematical references include [Menon's random matrix notes](https://www.dam.brown.edu/people/menon/publications/rmt-2021.pdf), [S-transform in finite free probability](https://arxiv.org/abs/2408.09337), and [unitary Hermite polynomials](https://arxiv.org/abs/2203.05533). Existing batch-only compound-Wishart/free-lognormal and continuous-gap illustrations remain separate; their analytical-branch and quadrature validation are outside these dashboards.

## Benchmarks

These small cases exercise all five benchmark tools. They write JSON only at the supplied output paths:

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

The 0.2.0 release includes the compatible symmetric S convention and [migration notes](https://github.com/jontb/FiniteFree/blob/main/CHANGELOG.md). Follow the [release review checklist](release.md) to inspect artifacts and checks on the final commit. The publish workflow uses `v*.*.*` tags or a manual dispatch and trusted publishing; invoking either is a separate release action.
