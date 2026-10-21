# Agent guidance for FiniteFree

FiniteFree is a Python library for finite free probability, random matrix
sampling and spectral computation. Agent-assisted contributions are welcome;
the contributor remains responsible for the submitted work.

## Start here

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for scope, review and branch policy.
- Use the [code map](docs/code-map.md) to find the relevant implementation and
  [API contracts](docs/api.md) for mathematical and numerical conventions.
- Check the issue discussion and existing PRs before working. Keep one issue or
  coherent change per PR. Discuss new APIs, changed conventions and substantial
  refactors before implementing them.
- Branch from current `develop`; contribution PRs target `develop`. `main` and
  the public documentation describe the published package.

## Setup and validation

Use Python 3.10 or later, a virtual environment and a C compiler. From the root:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install ".[dev]" Cython "setuptools<77"
python scripts/check.py
```

For fallback tests, the adversarial audit, executable examples and strict docs:

```bash
python -m pip install mkdocs-material "mkdocstrings[python]"
python scripts/check.py --full
```

Use focused checks while iterating; see [development commands](docs/development.md)
for individual checks and installed-wheel validation. Report exact commands,
results and checks not run. A missing dependency, skipped GPU check or smoke
benchmark is not evidence that the corresponding validation passed.

## Contracts to preserve

- Preserve public imports, signatures, explicit options, input/output ownership
  and legacy versus explicit RNG behavior unless a change was agreed.
- State dimension, normalization, domains and precision. Exact coefficients,
  certified root properties and floating-point estimates have different
  contracts; do not promote numerical diagnostics to certificates.
- Reproduce reported defects before fixing them. Use independent expected
  results: exact coefficients, analytic identities or a justified reference.
  Shared implementations agreeing with one another are insufficient alone.
- Follow [test contracts](tests/README.md). Strengthen an existing contract when
  practical; retain adversarial cases and independent oracles during consolidation.
  Do not loosen tolerances or remove assertions to make a failure disappear.
- Keep performance claims tied to measured inputs and environments. Ordinary
  correctness tests must not assert wall-clock speedups. Ask before launching
  expensive experiments beyond the agreed validation scope.

## Documentation and handoff

- Edit README and run `python scripts/sync_docs.py` to update `docs/index.md`.
  Do not hand-edit the generated index or commit ignored generated dashboards.
- Keep code, regression coverage and relevant docs together. Avoid unrelated
  formatting, dependencies, generated files and structural changes.
- Explain the resulting behavior, independent validation, compatibility effects,
  remaining limits and material agent assistance in the PR. Review the diff and
  verify cited references before handing it over.
- Community PRs require maintainer review and acceptance. Passing CI or an
  agent review does not authorize a merge. Do not merge, publish a release or
  change repository permissions or protections without explicit maintainer
  authorization; follow the separate [release process](docs/release.md).

These instructions guide work; GitHub permissions and repository rules control
access and merging. Contributors using a tool that does not load `AGENTS.md`
automatically should include it in their tool's project context.
