# Development and documentation

These instructions apply to the [v0.2.0 source](https://github.com/jontb/FiniteFree/tree/v0.2.0). Install the matching [PyPI release](https://pypi.org/project/finitefree/0.2.0/) with `python -m pip install finitefree==0.2.0`, or use a source checkout for development. A local build alone does not establish production publication. Record the Git commit when reproducing results.

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
