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

## Bounded concentration showcase

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python examples/gue_concentration.py --dimensions 16 32 64 --samples 2000 --seed 31841 --output /tmp/gue-concentration.json --plot-prefix /tmp/gue-concentration
```

This CPU example measures the fraction of GUE eigenvalues in `A=[-0.5,0.5]`.
A separate 20-matrix timing pilot preceded selecting the fixed 2,000-per-size
sample ceiling. Each invocation records its own separate pilot; pilot draws
are excluded, and sampling never stops based on the observed outcome. The CLI
limits dimensions to 64 and samples to 2,000 per size. A small CI smoke run uses
only eight matrices at dimension two, with no statistical pass/fail criterion.

The restricted Gram matrix is `G_ij = integral_A phi_i(x) phi_j(x) dmu_n(x)`,
where `dmu_n` is the same normalized Gaussian measure as above. Positive-weight
Gauss-Legendre rules of orders 96 and 192 produce numerical Gram matrices.
An example-local normalized basis recurrence supplies their factors; the
library's exact-verified projected-polynomial kernel independently checks the
integrated diagonal. Symmetry and spectrum in `[0,1]` are checked. Only spectral
violations within `1e-12` are clipped as roundoff, with the actual correction
recorded; material violations fail rather than being silently repaired.

By [HKPV, Theorem 7](https://arxiv.org/pdf/math/0503110), the interval count has
the law of a sum of independent Bernoulli variables with parameters equal to
the restricted operator eigenvalues. The **matrix eigenvalues themselves are
not independent**. The numerical reference computes the count PMF by Bernoulli
convolution and checks `m=tr(G)`, `v=tr(G-G²)`. Thus the fraction mean and variance
are `m/n` and `v/n²`. The two-sided Bernstein expression is
`min(1, 2 exp(-n² epsilon²/(2v+2n epsilon/3)))`. These identities and the inequality
apply to the exact operator; the plotted values plug in numerical quadrature.
They are **numerical finite-size theory, not certified bounds**. Refinement
differences and small roundoff corrections are not rigorous error enclosures.

`-variance.png` shows centered fraction masses and variance versus matrix size;
`-tails.png` shows empirical tails, pointwise two-sided 95% Clopper-Pearson
intervals, numerical count-law tails, and the Bernstein reference. Zero-hit
points are downward triangles at the one-sided 95% upper limit
`1-0.05**(1/R)`, never claimed zero probabilities. The intervals are not
simultaneous across thresholds. Variance error bars are `±1.96` estimated Monte
Carlo standard errors, not rigorous coverage guarantees. Each matrix is one
independent observation. Increasing `n` changes the fluctuation law; increasing
`R` only improves Monte Carlo estimation precision.

The JSON retains all observed counts, seeds, timing pilots, fixed sample counts,
source digests, Git state, dependency versions, CPU/thread settings, quadrature
and spectrum diagnostics, mean/variance disagreements and every tail interval.
It also compares `det(I-G)` with existing `gap_probability_continuous` on the
separately labeled, resolvable interval `[-0.5/n,0.5/n]`. It does not attempt to
resolve the extremely small full-window gap from eigenvalues rounded near one.
Expected-polynomial root fractions are reported separately and never used to
center the Monte Carlo statistic.

### Cloud smoke findings (2026-10-10)

The CPU run used seed 31841, 2,000 independent matrices per size, float64,
96/192-point quadrature and one BLAS thread on the AMD EPYC 9V45 cloud environment.
The preliminary pilot estimated less than one second of total matrix-sampling
work; the completed sampling took about 0.8 seconds (setup and theory excluded).
These are bounded-run observations, not performance claims.

| n | Numerical mean fraction | Numerical variance | Monte Carlo variance about numerical mean | Mean discrepancy / estimated SE |
| --- | --- | --- | --- | --- |
| 16 | 0.31505344 | 0.002010404 | 0.002025200 | +0.29 |
| 32 | 0.31495067 | 0.000550340 | 0.000555315 | +2.35 |
| 64 | 0.31495464 | 0.000155647 | 0.000158380 | +1.08 |

All three variance discrepancies were below 0.54 estimated standard errors.
The `n=32` mean lay outside a nominal `±1.96 SE` band; that disagreement is
retained rather than changing seeds or adding samples. None of the displayed
count-law tail references lay outside the pointwise binomial intervals, but
this is an observation, not validation of simultaneous coverage.

At `n=64, epsilon=0.04`, there were zero hits, while the numerical count-law
probability was about `0.000905` (roughly 1.8 expected hits). The zero-hit
one-sided upper limit is about `0.001497`. At `epsilon=0.08`, the numerical tail
is about `4.9e-11`, far beyond this simulation's resolution; the Bernstein
reference is about `0.00746` and is much looser than the count-law estimate.
The numerical mean is roughly constant while fraction variance shrinks by a
factor of 12.9 from `n=16` to `n=64`; this distinguishes concentration with matrix
size from increased Monte Carlo precision.

Maximum Gram refinement discrepancy was below `3.8e-14`, PMF refinement L1
change below `2.6e-13`, kernel/Gram mean discrepancy below `7.2e-15`, and the
resolvable gap disagreement below `4.5e-16`. Recorded negative eigenvalues
clipped as roundoff had magnitude below `5.2e-16`. These checks establish
numerical consistency on the measured cases, not certification of tiny tails.

- [x] CPU pilot, bounded simulation, deterministic identities and smoke test.
- [x] Reproducible report and inspected variance/tail plots.
- [ ] Rigorous quadrature enclosures or substantially rarer-tail validation.
- [ ] Real cloud GPU validation remains pending on the prepared-pencil track;
  this CPU mathematical showcase supplies no GPU evidence.
