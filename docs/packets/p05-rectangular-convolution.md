# P05: Rectangular coefficient convolution

**State:** planned. **Phase:** 2. **Roadmap:** T2.

**Feature issue:** [#45](https://github.com/jontb/FiniteFree/issues/45).

## Outcome

An exact rational rectangular additive convolution with a genuine dimension parameter.

## Prerequisites and decisions

[P00](p00-baseline.md) and an accepted rectangular dimension and coefficient contract.

- Freeze d by m inputs with m>=d, excess n=m-d and aspect ratio d/m; map these names explicitly to the [source's (n,d) convention](https://arxiv.org/html/2409.04305v1#S1.SS3).
- Specify whether lower-degree inputs are embedded or rejected. State monic normalization, rational ownership and the nonnegative-root domain of the preservation theorem.
- Keep the existing asymmetric_additive weights as root dilations. A rectangular parameter must not reinterpret that argument.

## Work and touchpoints

1. Implement the source coefficient formula using exact factorial/rational weights and existing polynomial storage.
2. Add a parameter dictionary and a small exactly expanded example. Preserve all existing convolution signatures and results.
3. Provide a small empirical rectangular Haar-sum workflow using the same d,m,n convention and actual sampled matrices. Keep its numerical/statistical comparison separate from the deterministic coefficient acceptance oracle.

Likely finitefree/convolutions.py and only agreed public exports; proposed tests/test_rectangular_convolution.py; docs/api.md, docs/tutorial.md and CHANGELOG.md.

## Independent acceptance

- Independently expand degrees 1–4 using the source's factorial formula, with asymmetric rational coefficients and more than one excess dimension.
- Verify the n=0 square-case relation against the current asymmetric convolution only after matching normalizations explicitly.
- Retain invalid-dimension, identity, symmetry, dilation, ownership and nonnegative-root examples. Shared implementation parity alone is insufficient.
- The empirical workflow records squared-singular-value versus symmetrized-spectrum conventions, zero atoms, expected-coefficient target and matrix-level uncertainty, preparing a usable counterpart for P06.

## Validation

Create the proposed module, then run:

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_rectangular_convolution.py tests/test_convolutions.py
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_rectangular_convolution.py tests/test_convolutions.py
```

Finish with profile R.

## Handoff and limits

Hand off exact coefficients and the dimension dictionary to P06. No rectangular transform, beta-deformed sum or changed legacy weights in this packet.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record) and the shared validation profiles. See the [roadmap](../roadmap.md) for the mathematical motivation.
