# P01: Exact normalized root moments

**State:** planned. **Phase:** 1. **Roadmap:** T1.

**Feature issue:** [#41](https://github.com/jontb/FiniteFree/issues/41).

P00 selected this next feature and prepared a
[concrete scope proposal](../p01-scope.md), including empirical input, ambient
dimension, pairing, output types and independently checked fixed examples.
It is not an accepted public API or an implementation.

## Outcome

An exact rational coefficient-to-root-moment operation and its supported empirical matrix/spectrum counterpart, with matching normalization and an executable comparison.

## Prerequisites and decisions

[P00](p00-baseline.md) and an accepted moment/dimension contract.

- Specify whether d is the degree or a larger ambient dimension with zero-root embedding. Define m_0=1, positive ambient dimension, scalar normalization and the constant-polynomial case.
- Specify accepted order/index inputs and rational result ownership. Orders above polynomial degree use Newton identities; they are not set to zero like finite cumulants above ambient dimension.
- Agree public naming and export placement before implementation; preserve the current finite-R API.
- Agree a direct numerical matrix/spectrum entry point or minimal adapter, including per-sample moments and averaging across independent matrices. Keep numerical error separate from sampling uncertainty.

## Work and touchpoints

1. Implement Newton-sum conversion from exact coefficients without numerically extracting roots. Reuse existing coefficient normalization rather than introducing a second incompatible convention.
2. Preserve invariance under a nonzero common polynomial scalar and account for shift/dilation and ambient zero roots.
3. Support empirical float64 moments from sampled Hermitian matrices or compatible spectra with the same dimension/zero-root convention. Use an explicit adapter if that keeps the algebraic API small; preserve existing comparison defaults.
4. Add a self-contained exact-versus-empirical tutorial/API example using the same samples; update the behavior changelog. Cache expansion and arbitrary-precision empirical evaluation remain separate scopes.

Likely finitefree/transforms.py and finitefree/__init__.py, with a small empirical adapter in an agreed existing module; proposed tests/test_root_moments.py and tests/test_empirical_transforms.py; docs/api.md, docs/tutorial.md and CHANGELOG.md. Touch core.py only if a small justified access change is needed.

## Independent acceptance

- For roots [1,3] at d=2, check m_1=2, m_2=5, m_3=14 and m_4=41 against direct rational power sums.
- At ambient d=4, use [0,0,1,3]: the positive-order moments halve and m_0 remains one. Test nonmonic coefficients, repeated/zero/negative roots and invalid domains.
- Use direct rational root lists or diagonal matrix traces as independent expectations. A round trip through the new implementation is only supplementary.
- Check the empirical path against direct normalized matrix-power traces and exact diagonal spectra. The two-sample aggregation case in the packet index must retain mean spectral moment 11 rather than substitute moment 9 from the mean characteristic polynomial. Include a small seeded ensemble comparison with matrix-level uncertainty.

## Validation

Create the proposed contract module, then run:

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_root_moments.py tests/test_empirical_transforms.py tests/test_nonmonic_normalization.py
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_root_moments.py tests/test_empirical_transforms.py tests/test_nonmonic_normalization.py
```

Finish with profile R and the executable documentation example.

## Handoff and limits

Hand off the paired algebraic/empirical contract and oracle table for P02. Empirical float64 support is required; arbitrary-precision empirical moments, extra measure transforms and new root-certification guarantees are separate scopes.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record) and the shared validation profiles. See the [roadmap](../roadmap.md) for the mathematical motivation.
