# P02: Full inverse finite-R reconstruction

**State:** planned. **Phase:** 1. **Roadmap:** T1.

**Feature issue:** [#42](https://github.com/jontb/FiniteFree/issues/42).

The [accepted scope](../p02-scope.md) records the public contract and
independently checked fixed examples. It is not yet implemented.

## Outcome

Recover a monic degree-d polynomial from complete finite-cumulant inputs through order d, preserving the exact rational path and supporting the corresponding empirical estimates with explicit numerical limits.

## Prerequisites and decisions

[P01](p01-exact-moments.md), P00's accepted baseline and an agreed reconstruction API.

- Freeze the current finite-R normalization and the mapping from cumulants to normalized coefficients. Specify exact input types, ambient dimension and output ownership.
- Keep this first packet to full reconstruction. Reject an incomplete sequence explicitly; never silently fill unspecified cumulants with zeros. If prefix reconstruction is required, scope its result representation as a separate follow-up.
- Algebraic reconstruction must not assert real-rootedness for arbitrary cumulants. Preserve lazy validation and explicit assumptions.
- Specify the numerical/precision contract for matrix-derived empirical cumulants. Distinguish reconstruction from per-sample cumulants, from averaged per-sample cumulants and from cumulants of estimated expected coefficients; these are different targets, and none identifies a unique matrix ensemble.

## Work and touchpoints

1. Invert the existing classical cumulant/normalized-coefficient recurrence with exact rational arithmetic and reconstruct coefficients using the accepted dimension convention.
2. Retain existing FiniteRTransform behavior, including its finite-order convention. Do not redirect legacy code to a new numerical inverse.
3. Document a complete convolution/cumulant/reconstruction example, with the distinction between coefficient equality and root geometry.
4. Provide the supported empirical input route and a forward/inverse example on the same small matrix samples. Label precision, conditioning and propagation of empirical uncertainty without treating floating observations as exact ensemble information.

Likely finitefree/transforms.py and finitefree/__init__.py; proposed tests/test_inverse_finite_r.py and the shared tests/test_empirical_transforms.py; existing tests/_cumulants.py references; docs/api.md, docs/tutorial.md and CHANGELOG.md.

## Independent acceptance

- Use independently expanded coefficients and the partition/formal-series reference. In the existing normalization, the degree-two polynomial [1,-4,3] has cumulants [2,2] and must reconstruct exactly.
- Check shifted, dilated, nonmonic-source and repeated/zero-root cases with independent values, plus a sequence whose polynomial fails real-rootedness.
- Round trips, additive convolution and scalar/power identities supplement fixed expected coefficients; none is the sole oracle. Invalid dimensions and incomplete inputs fail explicitly.
- Empirical reconstruction is checked against direct small-matrix determinant coefficients as well as the forward path. Retain the index's aggregation regression: per-draw second cumulants 2 and 2 differ from second cumulant 0 of the averaged polynomial. Numerical input conversion must not silently switch targets or claim real-rootedness.

## Validation

Create the proposed module, then run:

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_inverse_finite_r.py tests/test_empirical_transforms.py tests/test_transforms.py tests/test_cumulant_prefix_contracts.py
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_inverse_finite_r.py tests/test_empirical_transforms.py tests/test_transforms.py tests/test_cumulant_prefix_contracts.py
```

Finish with profile R.

## Handoff and limits

Hand off the reconstruction definition, exact and empirical examples and compatibility report. Arbitrary analytic-transform inversion, positivity inference and partial-sequence containers are excluded; the numerical route for empirical finite-cumulant inputs is required.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record) and the shared validation profiles. See the [roadmap](../roadmap.md) for the mathematical motivation.
