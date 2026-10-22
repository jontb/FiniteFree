# P07: Real/complex Jacobi–MANOVA workflow

**State:** planned. **Phase:** 2. **Roadmap:** E2.

**Feature issue:** [#47](https://github.com/jontb/FiniteFree/issues/47).

## Outcome

A MANOVA sampler at beta=1 or 2 and a small workflow connecting its spectrum with a correctly parameterized Jacobi reference.

## Prerequisites and decisions

[P00](p00-baseline.md) and an accepted real/complex MANOVA domain and output contract. P04 is not required.

- Freeze d,n1,n2 with nonsingular factors, initially n1,n2>=d. Specify real versus complex Gaussian draws, [0,1] support and output representation.
- Map dimension and beta to density exponents and shifted Jacobi parameters. Verify the expected-polynomial relation before adding any expectation factory.
- Choose a Hermitian matrix representation or a generalized Hermitian eigenproblem; do not label a nonsymmetric inverse product Hermitian.

## Work and touchpoints

1. Construct unnormalized Gram matrices A=XX* and B=YY*, and use stable solves/factorization for the accepted representation. If reusing normalized Wishart samples, explicitly undo their differing n1/n2 divisors.
2. Preserve source ownership and RNG isolation, and reject unsupported beta/rank domains explicitly.
3. Add a small expectation-polynomial example using the existing Jacobi family and independently derived parameters. A new expectation factory is included only if separately agreed within this same model.

Likely finitefree/ensembles.py and agreed exports; proposed tests/test_jacobi_manova.py; existing orthogonal tests; docs/api.md, a worked workflow and CHANGELOG.md.

## Independent acceptance

- For d=1, compare analytic moments with Beta(beta*n1/2,beta*n2/2), including mean n1/(n1+n2). Use the scalar distribution formula as an independent reference.
- For controlled small factors, compare Hermitian assembly with a separate generalized eigenvalue calculation and direct exact determinant coefficients.
- Check Hermitian symmetry, spectrum, parameter mapping, input/rng ownership and endpoint-adjacent cases. A histogram or range check alone is insufficient.
- The worked transform comparison uses the actual sampled matrix/spectrum with matching normalization and an explicit per-sample or ensemble target; mean sampled eigenvalues and expected-polynomial roots are not interchangeable.

## Validation

Create the proposed module, then run:

```bash
PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_jacobi_manova.py tests/test_ensembles.py tests/test_orthogonal.py
PYFFP_DISABLE_CYTHON=1 PYTHONPATH=. python -m pytest --import-mode=importlib tests/test_jacobi_manova.py tests/test_ensembles.py tests/test_orthogonal.py
```

Finish with profile R.

## Handoff and limits

Hand off the MANOVA/Jacobi parameter dictionary and exact examples. General beta, endpoint atoms from rank-deficient factors, and a generic covariance API are outside this sampler's scope.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record) and the shared validation profiles. See the [roadmap](../roadmap.md) for the mathematical motivation.
