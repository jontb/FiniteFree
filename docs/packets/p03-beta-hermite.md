# P03: Compact beta-Hermite sampling

**State:** implemented as `sample_beta_hermite`. **Phase:** 1. **Roadmap:** E1.

**Feature issue:** [#43](https://github.com/jontb/FiniteFree/issues/43).

The [accepted scope](../p03-scope.md) records the model, scale, compact output
and RNG contract with independently checked identities.

## Outcome

A general-beta Hermite tridiagonal sampler with explicit storage and normalization.

## Prerequisites and decisions

[P00](p00-baseline.md) and an accepted density, scale, compact-output and RNG contract. P01/P02 are not mathematical prerequisites.

- Use the [Dumitriu–Edelman model](https://arxiv.org/abs/math-ph/0206043), deriving the conversion to the library's order-one spectral scale. State beta>0, dimensions and the admissible scale domain.
- Choose one compact representation for diagonal/off-diagonal arrays and any explicit conversion to dense form. Eigenvalue output must be a deliberate choice, not a hidden dense allocation.
- For new calls, specify explicit Generator ownership and the behavior when rng is omitted. Existing GOE/GUE/GSE samplers and draw order remain unchanged.

## Work and touchpoints

1. Implement the accepted normal/chi construction, validating parameters before consuming randomness.
2. Retain owned output arrays and O(d) compact storage. Add a small matched classical-model example and explain spectral-law equivalence versus different entry laws.
3. Derive the expectation-polynomial recurrence rather than estimating its coefficients as the only reference.
4. Include an end-to-end transform example on the actual compact/sampled output, with the same spectral scale and a declared per-sample or ensemble target. Use the transform's supported empirical route rather than substituting roots of the expectation polynomial.

Likely finitefree/ensembles.py and finitefree/__init__.py; proposed tests/test_beta_hermite.py; existing ensemble/RNG contracts; docs/api.md and CHANGELOG.md.

## Independent acceptance

- Independently verify normal/chi parameters and moment identities. Under the proposed scale matching classical GOE/GUE, E[Tr(H^2)/d]=1+(2/beta-1)/d; derive and check that identity before freezing the convention.
- Use controlled draws for exact construction/shape checks, d=1 analytic laws, noninteger beta, invalid inputs, ownership and RNG isolation.
- Report a bounded seeded spectral comparison at beta values 1, 2 and 4, using one eigenvalue per quaternionic pair. Do not demand identical matrices, identical draws across algorithms or a single must-pass random p-value.

## Validation

Create the proposed module, then run:

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_beta_hermite.py tests/test_ensembles.py
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_beta_hermite.py tests/test_ensembles.py
```

Finish with profile R. Any larger law study uses a separately agreed E protocol.

## Handoff and limits

Hand off the compact-output/RNG conventions to P04. No replacement of dense samplers, GPU implementation or general backend framework.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record) and the shared validation profiles. See the [roadmap](../roadmap.md) for the mathematical motivation.
