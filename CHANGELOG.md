# Changelog

## Unreleased

- Proper polynomial projections compute the leading `j+1` coefficients directly rather than constructing `d-j` intermediate derivatives. Exact coefficients and monic normalization are retained, including nonmonic inputs. Known Hermite projections preserve variance, center and numerical recurrence provenance. Full-degree projections retain object identity. Invalid noninteger or out-of-range dimensions raise `ValueError`.

- Finite free cumulants use `(-d)**(n-1) * c_n / (n-1)!`, following Definition 2.14 of Arizmendi et al. Earlier code multiplied by `(n-1)!`. For orders `n >= 3`, recompute saved cumulants or convert with `new = old / ((n-1)!)**2`. The first two cumulants are unchanged. The matching additive-power inverse is updated, preserving valid polynomial coefficients.
- Reconstructed normalized coefficient sequences and Jacobi/Laguerre polynomials outside proven orthogonality domains are validated lazily. Root extraction and positive-root domain checks reject complex-rooted inputs rather than accepting false certificates. Inability to certify raises `RuntimeError`.
- DPP sampling validates real symmetric correlation kernels, spectrum, shape and finiteness. Invalid kernels raise `ValueError`; numerical roundoff up to `1e-10` is allowed. Kernel subsets/permutations respect the requested state space.
- Stable tridiagonal root evaluation is available for classical orthogonal families, GUE/Wishart expectations, affine transforms and proven Hermite additive convolutions with `exact=False`. The default `exact=True` path is retained, and reference requests bypass approximate caches. Positive integer root powers preserve exact coefficients.

For stored cumulants from the earlier implementation:

```python
from math import factorial

saved_old_cumulants = [1, 3, 36]
migrated = [value / factorial(n - 1)**2
            for n, value in enumerate(saved_old_cumulants, start=1)]
assert migrated == [1, 3, 9]
```
