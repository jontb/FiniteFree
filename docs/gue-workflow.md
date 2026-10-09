# Composing the GUE spectral workflow

The development example composes existing functionality on the sampler's common
variance scale `1/d`:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python examples/gue_spectral_workflow.py --dimension 8 --samples 256 --output /tmp/gue-workflow.json
# Optional Matplotlib figure:
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python examples/gue_spectral_workflow.py --dimension 8 --samples 256 --plot /tmp/gue-workflow.png
```

It constructs `p = gue_expected_poly(d)`, obtains the entire compatible basis as
`[p.projection(j) for j in range(d+1)]`, and uses exact norms `j! / d**j` with
`OrthogonalPolynomialKernel`. Every degree retains variance `1/d`; independently
constructing a different GUE dimension for each degree would change that measure.
The reference measure is the normalized Gaussian
`w_d(x) dx = sqrt(d/(2*pi)) exp(-d*x*x/2) dx`.

The mean ESD density is `w_d(x)*K(x,x)/d`. Existing `sample_gue(d)` matrix draws
feed `EmpiricalComparison(p, samples, generator)` once, and its retained
eigenvalues supply pooled ESDs, per-matrix second moments and gap frequencies.
The JSON includes expected and sampled characteristic coefficients, the existing
coefficient confidence diagnostic, and second-moment standard errors computed
across independent matrices. The optional figure overlays sampled ESDs and mean
density with separately labeled expected-polynomial-root markers. No grid-based
DPP sampling is substituted for the continuous GUE matrix sampler.

Mean ESD and expected-polynomial-root measures differ at finite size: their
second moments here are `1` and `(d-1)/d`, respectively. Regression checks use
exact recurrence/Newton identities and Gaussian quadrature for mass and moments;
statistical outcomes are diagnostics rather than pass/fail assertions. For the
same Gaussian weight, `gap_probability_continuous` at 40 and 80 quadrature points
is compared with the matrix-sample gap frequency and its Wilson interval. The
quadrature refinement difference is a numerical diagnostic, not certification.

Internally, kernels verify rational monic recurrence, supplied leading
coefficients and monic squared norms before activating a normalized Hermite
recurrence. A common center, positive variance, positive measure mass and signed
basis rescaling are supported. Exact ratios precede float conversion; large
factorials are never converted separately. Exact evaluations and generic family
fallbacks retain their existing paths and signatures. Root provenance alone
cannot determine the measure's mass, so the kernel verifies its owned inputs.
The construction follows the [orthogonal-polynomial recurrence and kernel
identities](https://dlmf.nist.gov/18.2) and the
[orthogonal-polynomial ensemble framework](https://arxiv.org/abs/1709.01287).

### Subsequent concentration example checklist

- [ ] Pilot a bounded CPU simulation for the fraction of GUE eigenvalues in
  `[-0.5,0.5]` at dimensions 16, 32 and 64; select sample count after timing
  (2,000 per size is a ceiling, not a requirement).
- [ ] Form the restricted orthonormal-basis Gram matrix with positive-weight
  quadrature. Check symmetry, spectrum in `[0,1]`, and quadrature refinement;
  compare its trace with integrated kernel density and determinant gap with
  `gap_probability_continuous` on a resolvable interval.
- [ ] Use the Bernoulli-sum count law to compute numerical finite-size theory:
  `m=tr(G)`, `v=tr(G-G²)`, fraction mean `m/d`, variance `v/d²`, and two-sided
  Bernstein bound `min(1,2 exp(-d² epsilon²/(2v+2d epsilon/3)))`. This is a law
  for the count; matrix eigenvalues are not independent. See
  [HKPV, Theorem 7](https://arxiv.org/abs/math/0503110).
- [ ] Treat each matrix as one independent observation. Show binomial
  uncertainty for estimated tails; zero exceedances has one-sided 95% upper
  limit `1-0.05**(1/R)`, not proven zero probability. Distinguish increasing
  matrix size from increasing Monte Carlo replicate count. Center on the
  finite-size mean density, not polynomial roots or the limiting semicircle.
- [ ] Label quadrature-based overlays as numerical finite-size theory until
  rigorously enclosed. Keep this a separate example, with no new public API.
