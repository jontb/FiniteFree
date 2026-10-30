# P04: Compact beta-Laguerre sampling

**State:** implemented as `sample_beta_laguerre`. **Phase:** 1. **Roadmap:** E1.

**Feature issue:** [#44](https://github.com/jontb/FiniteFree/issues/44).

The [accepted scope](../p04-scope.md) records the parameter dictionary,
expected polynomial and compact-output contract with independently checked
identities.

## Outcome

A compact general-beta Laguerre/Wishart spectral model using the same explicit output and RNG conventions as P03.

## Prerequisites and decisions

[P03](p03-beta-hermite.md) and an accepted Laguerre shape/normalization dictionary.

- Derive the relation between shape, beta, chi degrees of freedom, density exponent and matrix normalization from the source model. State the full domain rather than copying integer Wishart restrictions onto a real shape parameter.
- Distinguish a bidiagonal factor from its positive tridiagonal Gram matrix and from returned eigenvalues. State how real shape relates to the existing integer sample-size n.
- Specify which expectation-polynomial parameters are exact rationals and which calculations are floating-point estimates.

## Work and touchpoints

1. Implement one accepted compact model; validate every chi degree of freedom and scale before draws. Preserve existing sample_wishart and wishart_expected_poly behavior.
2. Add a parameter dictionary and a small comparison at beta values 1, 2 and 4 with existing Wishart spectra, respecting paired eigenvalues.
3. Provide an independently derived low-degree characteristic-polynomial recurrence and one executable workflow.
4. Connect the sampled compact/Gram output to a supported empirical transform using the same beta/shape, squared-spectrum and zero-root conventions, with explicit aggregation order and numerical/statistical limits.

Likely finitefree/ensembles.py and finitefree/__init__.py; proposed tests/test_beta_laguerre.py; relevant orthogonal-family tests; docs/api.md and CHANGELOG.md.

## Independent acceptance

- Check chi-square means/variances and scalar gamma laws directly. Use controlled draws and a separate small matrix product to validate compact assembly.
- Test noninteger beta/shape, parameters near the admissible boundary, invalid degrees of freedom, ownership, empty/invalid shape policy and RNG isolation.
- Compare exact low-degree coefficients with the derived Laguerre recurrence. Classical-model Monte Carlo and root plots remain supplementary.

## Validation

Create the proposed module, then run:

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_beta_laguerre.py tests/test_beta_hermite.py tests/test_ensembles.py tests/test_orthogonal.py
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_beta_laguerre.py tests/test_beta_hermite.py tests/test_ensembles.py tests/test_orthogonal.py
```

Finish with profile R; separate any larger E study from ordinary correctness tests.

## Handoff and limits

Hand off the accepted beta/shape dictionary and exact coefficient examples. General-beta Jacobi, singular endpoint extensions and new covariance wrappers are excluded.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record) and the shared validation profiles. See the [roadmap](../roadmap.md) for the mathematical motivation.
