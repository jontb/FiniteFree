# Code map and contribution boundaries

Public imports are collected in `finitefree/__init__.py` and
`finitefree/hyperbolic/__init__.py`. Preserve those import paths when changing
internals. The current package layout remains in place; discuss larger splits
before moving code or changing ownership of cached state.

| Area | Sources | Contract to preserve |
| --- | --- | --- |
| Polynomial objects and roots | `core.py`, `utils/roots.py`, `utils/prs.py` | Exact coefficient storage, lazy real-root certification, owned inputs, cache precision and numerical output limits |
| Convolutions and transforms | `convolutions.py`, `transforms.py` | Coefficient normalization, ambient dimension, transform conventions, domains and explicit options |
| Orthogonal families | `orthogonal.py` | Family parameters, normalization, exact recurrences and supported certification domains |
| Multivariate polynomials | `multivariate.py` | Ordered symbol contexts, exact sparse algebra, derivative shapes, owned exports and explicit numerical limits |
| Matrix pencils and SLPs | `hyperbolic/pencils.py`, `hyperbolic/slp.py` | Rational input preservation, determinant conventions and the documented singular-point derivative limits |
| Modular reconstruction | `utils/modular.py`, `utils/modular_fast.pyx`, `utils/parallel.py` | Integer reduction before narrowing, valid prime-field interpolation, coefficient bounds and compiled/fallback agreement |
| Random matrices and empirical comparisons | `ensembles.py`, `streaming.py` | Ensemble scaling, paired spectra, legacy/explicit RNG behavior, bounded online storage and the statistical scope of coefficient diagnostics |
| DPP kernels and sampling | `dpp.py` | Kernel domains, reference measure and normalization, owned preparation, state order and sampling law |
| Optional numerical kernels and count laws | `numerical.py` | Normalized owned basis, explicit measure density, reused quadrature, directly summed tails and estimated versus certified probabilities |
| Precision and conversion | `utils/precision.py`, `utils/conversion.py` | Restored precision context, supplied binary float values and explicit output range/rounding limits |

Module paths in the table are relative to `finitefree/`. Library changes belong
with tests in `tests/`. Use independent expected results for numerical fixes,
and exercise both source acceleration and documented fallback paths.

## Tooling, examples and documentation

- `scripts/check.py` runs routine checks; `scripts/tests/` covers source-only
  tools and example generators. Packaging helpers verify installed wheels away
  from the checkout and inspect their native extensions.
- `examples/` contains runnable public-API workflows and development HTML-data
  generators. Examples explain conventions and limitations; an illustration is
  not a mathematical certificate.
- `visuals/interactive/` contains editable browser templates and interaction
  checks. `visuals/generated/` contains ignored outputs. The batch scripts in
  `visuals/` have separate optional plotting dependencies and can be expensive.
- README is the source for generated `docs/index.md`. API, tutorial,
  multivariate, development and release prose live in `docs/`.
- `.github/workflows/` separates routine CI, documentation deployment, artifact
  builds and publication. Keep required check names stable when editing CI;
  release tags and publishing dispatches are separate release actions.

See the [development commands](development.md),
[API contracts](api.md) and [release process](release.md). The
[contribution guide](https://github.com/jontb/FiniteFree/blob/develop/CONTRIBUTING.md)
describes branch, review and evidence expectations.
