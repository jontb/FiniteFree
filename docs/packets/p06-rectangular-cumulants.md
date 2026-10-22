# P06: Rectangular cumulants and finite R

**State:** planned. **Phase:** 2. **Roadmap:** T2.

**Feature issue:** [#46](https://github.com/jontb/FiniteFree/issues/46).

## Outcome

Exact rectangular cumulants and their finite-R representation, plus a usable empirical rectangular matrix/spectrum path with the same conventions and demonstrated algebraic linearization of P05.

## Prerequisites and decisions

[P05](p05-rectangular-convolution.md) and a reviewed rectangular moment-cumulant definition.

- Use the same d,m,n and aspect-ratio dictionary as P05. Fix transform indexing, signs, truncation and any normalization factors from the primary definition.
- Distinguish moments of squared singular values, symmetrized singular values and the additional zero eigenvalues of the larger Gram matrix.
- Reuse accepted T1 output conventions where available, but do not require unmerged P01/P02 code or modify their scalar finite-R meaning.
- Specify the empirical input/aggregation contract for rectangular matrices or supplied squared singular values, including matrix-level uncertainty, aspect ratio and larger-Gram zero atoms. Reuse a small common adapter when appropriate.

## Work and touchpoints

1. Implement the exact rectangular moment-cumulant recurrence and agreed transform output with rational arithmetic, alongside the required numerical empirical route with declared conditioning/range limits.
2. Add an independently checkable low-degree table and a convolution/transform example.
3. Add the required small exact-versus-empirical rectangular Haar-sum example using actual sampled matrices. State whether the transform is applied before averaging or to estimated expected coefficients; its Monte Carlo comparison supplements the exact oracle.

Likely finitefree/transforms.py and agreed exports; proposed tests/test_rectangular_cumulants.py and the shared tests/test_empirical_transforms.py; P05 regressions; docs/api.md and CHANGELOG.md.

## Independent acceptance

- Independently evaluate low-order coefficients with a direct source moment-cumulant or partition formula rather than calling the new recurrence.
- Check exact additivity under P05 for rational examples, plus dilation, square-case conventions, zero roots and domain failures.
- Do not assume the square rectangular transform equals the existing symmetric finite R without deriving the map. Any limiting comparison must keep aspect ratio and normalization explicit.
- Use exact small rectangular diagonal/singular-value examples and independent Gram traces/SVD checks for empirical outputs. Check normalization, zero-atom handling and aggregation order; a transform of averaged sample coefficients must not be substituted for an average of sample transforms.

## Validation

Create the proposed module, then run:

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_rectangular_cumulants.py tests/test_empirical_transforms.py tests/test_rectangular_convolution.py tests/test_transforms.py
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_rectangular_cumulants.py tests/test_empirical_transforms.py tests/test_rectangular_convolution.py tests/test_transforms.py
```

Finish with profile R. A larger Haar study requires profile E and its own budget.

## Handoff and limits

Hand off the definition-to-code correspondence, exact tables and required empirical matrix path/example. Numerical transform inversion, beta-deformed rectangular addition and asymptotic-performance claims are excluded.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record) and the shared validation profiles. See the [roadmap](../roadmap.md) for the mathematical motivation.
