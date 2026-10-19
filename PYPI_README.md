# FiniteFree

Python tools for finite free probability, random matrix sampling and spectral
computation. FiniteFree focuses on efficient repeated calculations, numerical
stability and reproducible workflows, with explicit precision and storage
contracts.

## Install

```sh
python -m pip install finitefree
```

Requires Python 3.10 or later. Release wheels cover CPython 3.10–3.13 on Linux
x86-64, macOS ARM64 and Windows x86-64. A compatible wheel avoids compiling
FiniteFree locally; source builds require a C compiler. On musl Linux,
python-flint may also require FLINT/GMP/MPFR source-build prerequisites.

Start with the [tutorial](https://jontb.github.io/FiniteFree/tutorial/) or the
[API reference](https://jontb.github.io/FiniteFree/api/).

## Main workflows

- **Polynomial calculations:** finite free convolutions, cumulants, transforms
  and classical orthogonal families, with integer/rational coefficient arithmetic
  where supported and numerical root evaluation.
- **Reproducible sampling:** Gaussian, Wishart and Haar random matrices, plus
  discrete DPP sampling. Pass an optional NumPy `Generator` through `rng=`;
  omitted arguments preserve the existing global RNG behavior.
- **Reusable numerical work:** prepared orthogonal kernels, batched basis/kernel
  evaluation and reusable interval quadrature; prepared matrix pencils reuse LU
  factorizations for log-determinants, gradients and Hessian-vector products.
- **Count statistics:** finite-rank DPP count PMFs, CDFs, directly summed upper
  tails, means and variances.
- **Bounded-storage summaries:** streaming empirical coefficient moments and
  configured interval-count histograms. Retained-sample comparisons remain
  available when later plots or individual observations are needed.

See the [GUE spectral workflow](https://jontb.github.io/FiniteFree/gue-workflow/)
for a worked example connecting sampling, expected polynomials and kernels.
Source-checkout examples and plotting dependencies are documented separately.

## Precision and performance

Exactness applies to supported integer/rational algebraic operations. Roots,
sampled matrices, prepared numerical kernels and continuous probabilities use
floating-point methods with precision and conditioning limits. An `exact=True`
root request still returns float64 values, rather than exact roots or certified
error bounds.

Continuous gap and count probabilities are quadrature estimates. Supply the
kernel's reference measure density explicitly for weighted measures, including
Hermite kernels. Refinement and spectrum diagnostics do not certify rare tails.

Performance depends on the input, chosen API and environment. Prepared objects
expose reuse; streaming summaries avoid retaining every sample. GPU correctness
checks do not establish a speed or memory advantage. See the
[numerical contracts](https://jontb.github.io/FiniteFree/api/#optional-numerical-workflows)
and [benchmark instructions](https://jontb.github.io/FiniteFree/development/#benchmarks).

## Release and project links

Version 0.4.1 updates package descriptions, project links and documentation.
Library calculations, defaults and runtime dependencies are unchanged from 0.4.0.

- [Documentation](https://jontb.github.io/FiniteFree/)
- [Source and contributions](https://github.com/jontb/FiniteFree)
- [Issue tracker](https://github.com/jontb/FiniteFree/issues)
- [Changelog and compatibility notes](https://github.com/jontb/FiniteFree/blob/v0.4.1/CHANGELOG.md)

MIT licensed.
