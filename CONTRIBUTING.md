# Contributing to FiniteFree

Contributions are welcome: reproducible bug reports, documentation, tests,
numerical improvements and mathematical extensions. Search existing issues and
pull requests before starting. Discuss new public APIs, changed mathematical
conventions and substantial refactors in an issue first; small fixes can go
straight to a pull request.

## Branches and pull requests

Start a short-lived branch from the current `develop` branch and open your pull
request against `develop`, including for maintainer-authored changes. Fork the
repository if you do not have push access. Choose a branch name that describes
one coherent change, and use draft PRs for work that needs discussion.

Keep fixes, regression tests and related documentation together. Prefer separate
commits for independently reviewable changes. Avoid unrelated formatting or
generated assets. Describe the problem, resulting behavior, validation and
remaining limits in the PR template; link an existing issue when applicable.

Required CI checks must pass. The current single-maintainer policy requires
**zero approving reviews** and permits the maintainer to merge their own PRs.
Peer review is encouraged when available; there are no mandatory code-owner
approvals. Request a fresh review after substantive changes and resolve concrete
review concerns before merging.

`main` and the public documentation track the published package. Release tags
publish packages, so release approval, publication and promotion to `main` follow
the separate [release process](docs/release.md). A contribution PR does not
authorize a release.

## Local setup and checks

Use Python 3.10 or later and a C compiler. From a checkout of your branch:

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install ".[dev]" Cython "setuptools<77"
python scripts/check.py
```

The check runner uses the active Python interpreter, builds the source Cython
extension and runs Ruff lint/format checks, strict mypy and the source/tooling
test suites. It works from any working directory when invoked by its path,
stops at the first failure and leaves the calling process's environment intact.
It installs no dependencies and changes no tracked source files.

For fallback tests, examples, README/index consistency and strict documentation:

```bash
python -m pip install mkdocs-material "mkdocstrings[python]"
python scripts/check.py --full
```

After a standard installation of the current candidate, add `--installed-wheel`
to test the installed package outside the source tree. Reinstall after source
changes; an earlier installation does not validate the current candidate.
`PYFFP_DISABLE_CYTHON=1` selects the documented grid/CRT fallback paths, rather
than simulating an extension-free installation. See the
[development guide](docs/development.md) for individual commands, pinned minimum
dependencies, browser checks, benchmarks and release artifact validation.

## Mathematical and numerical review

- State the invariant or documented contract being preserved. Include dimensions,
  normalization, domains and precision when they affect the result.
- Reproduce a bug before fixing it and add a regression with an independent
  expected result: exact coefficients, a separately expanded determinant, an
  analytic identity or a justified high-precision reference. Agreement between
  two paths sharing the same implementation is insufficient by itself.
- Preserve public signatures, explicit options, ownership and RNG behavior unless
  an API change has been discussed. Document compatibility or migration effects.
- Distinguish exact identities and certified results from numerical diagnostics
  and finite-sample evidence. Report unresolved discrepancies and limits.
- Keep performance measurements reproducible, with commit, versions, hardware,
  seeds, input sizes and the scope of timing. Avoid timing thresholds in ordinary
  correctness tests.

The [code map](docs/code-map.md) identifies the modules and their boundaries.
Edit README and regenerate `docs/index.md` together using
`python scripts/sync_docs.py`; do not hand-edit the generated index. Keep
unreleased behavior documented with its development source.

## AI-assisted contributions

AI-assisted work is welcome. Disclose material assistance in the PR and review
the resulting code, references and tests yourself. Contributors remain
responsible for correctness and reproducibility; generated claims need the same
independent evidence as other contributions.
