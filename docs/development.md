# Development and documentation

These instructions target the unreleased source checkout. [PyPI 0.1.0](https://pypi.org/project/finitefree/0.1.0/) corresponds to tag [v0.1.0](https://github.com/jontb/FiniteFree/tree/v0.1.0). The source metadata still reads `0.1.0`; an installed development wheel can therefore report the same version as that release. Record the Git commit when reproducing development results.

## Installation

Use the [README setup instructions](index.md#installation) to create a virtual environment. The declared minimum is Python 3.9. CI is configured for CPython 3.9–3.13 on Linux, macOS and Windows; verify the matrix on the intended release commit. Dependency minima are declarations, not a separately tested minimum-dependency matrix.

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

The installed-wheel helper runs library tests in a temporary directory and confirms that the compiled extension is present. Documentation-tool tests require the source tree. `PYFFP_DISABLE_CYTHON=1` selects the Python modular fallback; it does not remove the built extension. In PowerShell, set the corresponding variables with `$env:PYTHONPATH="."` and `$env:PYFFP_DISABLE_CYTHON="1"`, then clear them before installed-wheel checks.

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

`visuals/*.py` regenerate pre-rendered illustrations and require Matplotlib/Pillow. They may write many frames and use high-degree solvers or approximate grid sampling; their output is illustrative rather than part of the regression suite.

## Preparing 0.2

No 0.2 package or tag is created by these instructions. Before a release, integrate the reviewed fixes, run the complete supported-platform matrix on that commit, finalize [migration notes](https://github.com/jontb/FiniteFree/blob/main/CHANGELOG.md), choose the symmetric S-output convention, update the version metadata and documentation scope, and inspect the built wheel/sdist metadata. The publish workflow uses `v*.*.*` tags or a manual dispatch and trusted publishing; invoking either is a separate release action.
