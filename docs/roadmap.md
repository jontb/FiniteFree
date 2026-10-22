# Feature roadmap: Phases 1 and 2

**Planning snapshot: 2026-10-21, America/Toronto.**
This is a proposed development sequence. Milestones depend on mathematical
review, independent validation and maintainer capacity. Public API names and release versions remain undecided.

This public plan covers Phase 1 transforms and beta ensembles, and Phase 2
rectangular and Jacobi models. Each contribution should focus on one sampler or
transform, including its empirical matrix counterpart. Independent contributions
can proceed in parallel once their shared conventions are agreed.

Use the [phased implementation packets](implementation-packets.md) for concrete
PR scopes, prerequisites, independent acceptance and validation commands. The
[contribution issues](implementation-packets.md#claimable-phase-1-and-2-features)
let contributors take on individual samplers and transforms.

The [P00 baseline report](p00-baseline-report.md) is ready for review, with a
[concrete P01 scope proposal](p01-scope.md). The audit assigned existing empirical
Cauchy/S/T/symmetric-S gaps to [P16](packets/p16-empirical-legacy-transforms.md),
which has a separate contribution issue for each transform.

**Required throughout:** transforms must support their empirical random-matrix
counterparts. Each transform packet must deliver a usable matrix/spectrum path,
matching dimension and scaling conventions, independent validation and a worked
exact-versus-empirical example. This includes inverse operations; empirical
support is part of feature completion.

## Starting point

The baseline is [`develop` at `2e13bc2`](https://github.com/jontb/FiniteFree/tree/2e13bc2832c5643df9c9d2730f5721f3b205121d).
The latest GitHub release is [0.4.1](https://github.com/jontb/FiniteFree/releases/tag/v0.4.1).
The [code map](code-map.md) and [API contracts](api.md) describe the current scope:

- Exact rational polynomial algebra and three coefficient convolutions;
  finite Cauchy, R, S, T and symmetric S transforms.
- GOE/GUE/GSE, Wishart and Haar samplers; Gaussian/Wishart expectation
  polynomials; retained and streaming empirical comparisons.
- Prepared matrix pencils, Hermite-aware numerical kernels, reusable
  quadrature and finite-rank DPP count laws.
- Compiled/fallback regression contracts, an adversarial audit, installed-wheel
  validation and the bounded [numerical-v1 benchmark protocol](benchmark-plan.md).

Build on these capabilities. Prepared-quadrature count-law caching is already on
`develop`; CRT reconstruction and callable-DPP complex-input fixes are already
documented in the [0.3.0 changelog](https://github.com/jontb/FiniteFree/blob/2e13bc2832c5643df9c9d2730f5721f3b205121d/CHANGELOG.md#030).
Keep their regression coverage instead of planning duplicate fixes.

The live backlog needs reconciliation: [#35](https://github.com/jontb/FiniteFree/issues/35)
and [#36](https://github.com/jontb/FiniteFree/issues/36) remain open, although their
implementations were merged in [#39](https://github.com/jontb/FiniteFree/pull/39)
and [#40](https://github.com/jontb/FiniteFree/pull/40).
The finite-state DPP tutorial [#37](https://github.com/jontb/FiniteFree/issues/37)
remains a useful scoped documentation task. Recheck discussions, assignments and
PRs before starting any item; this document creates no assignment.

## Priorities and sequence

| Milestone | Main deliverable | Exit gate |
| --- | --- | --- |
| M0: consolidate the baseline | Reconcile existing work, record open numerical limits, finish scoped examples | A current evidence inventory and agreed first feature scopes |
| M1: transforms and beta ensembles | Exact moment/cumulant conversions and inverse finite R; beta-Hermite, followed by beta-Laguerre | Independent algebraic and sampling references, documented normalization |
| M2: rectangular and bounded spectra | Rectangular convolution/cumulants; Jacobi/MANOVA sampling | Matched dimension conventions and exact low-degree references |

Use each feature's actual prerequisites rather than requiring an entire phase
to finish. Keep scope focused and independent validation intact.

## A. New transforms and inverse operations

### Required empirical matrix counterparts

Plan paired algebraic and empirical interfaces for the existing Cauchy, R, S,
T and symmetric S transforms and for every new transform. Empirical inputs may
use a small shared matrix/spectrum adapter instead of duplicating eigensolvers.
Cover the appropriate Hermitian, positive, rectangular/singular-value or unitary
domain; compact ensemble outputs must have a compatible route. Record existing
support gaps in P00 and include their resolution in the affected transform scope.

Every interface must name its target: a transform of each sample's characteristic
polynomial/spectral measure, a transform of the estimated expected characteristic
polynomial, or a transform of the estimated mean spectral measure. State the order
of averaging and transformation. These targets can differ; finite cumulants are
also distinct from classical free cumulants of an empirical spectral measure.
For inverses, support empirically estimated inputs with explicit precision,
conditioning and uncertainty limits. Reconstruction does not identify a random
matrix ensemble uniquely.

Use the same samples and conventions for exact/empirical comparisons, retain
analytic or exact matrix references, and report sampling uncertainty separately
from numerical error. Respect positivity, symmetry, zero-root, pairing, branch
and pole restrictions. Invalid observations must be reported or rejected
explicitly, rather than silently discarded or projected into the required domain.
See the [packet-level requirement](implementation-packets.md#required-empirical-matrix-counterparts).

### T1: exact moments and inverse finite R — first priority

Support a complete, explicit chain between polynomial coefficients, normalized
root moments and finite free cumulants. Add inverse finite-R reconstruction so
users can specify cumulants, recover coefficients and study convolution powers
without hand-written conversion code. Use the existing finite-R normalization;
compare its convention with the original
[finite cumulant and moment-cumulant theory](https://arxiv.org/abs/1611.06598).

The contract must state ambient dimension, monic normalization, coefficient
order and truncation. A full sequence through degree `d` determines coefficients;
a shorter prefix determines only a coefficient prefix. Arbitrary cumulants can
produce a polynomial that is not real-rooted, so algebraic reconstruction and
geometric validation must remain separate. Rational inputs should retain exact
rational arithmetic. Numerical reconstruction needs explicit precision limits.
Specify the zero-root embedding and moment normalization when polynomial degree
is below the ambient dimension.

Deliver numerical moments/finite cumulants from matrix samples alongside the
coefficient path, and allow reconstruction from the corresponding empirical
cumulant inputs. Preserve the finite-degree convention in both paths; label
per-sample and ensemble estimates and validate them independently.

Acceptance requires small independently expanded rational polynomials, Newton
identities and the existing partition/formal-series oracles. Include nonmonic
inputs, different ambient dimensions, shifts, dilations, repeated/zero roots and
invalid prefixes. Round trips and convolution additivity supplement those
references; they cannot be the only checks.

### T2: rectangular finite R and cumulants — second priority

Introduce the rectangular convolution and its cumulants together, with an
explicit dimension pair. A useful convention to review is a `d` by `m` matrix,
`m >= d`, with excess dimension `n = m-d` and aspect ratio `d/m`. Document how
this maps to the source's `(n,d)` notation. The
[rectangular cumulant theory](https://arxiv.org/html/2409.04305v1#S1.SS3)
provides convolution formulas and linearizing cumulants for this setting.

Distinguish squared singular values of `AA*`, symmetrized singular values and
the extra zero eigenvalues of the larger Gram matrix. The current
`asymmetric_additive(..., weights=...)` uses weights to dilate inputs; those
weights do not encode rectangular dimensions. Preserve that API and explicitly
verify the square-case relation before sharing an implementation.

Acceptance requires direct low-degree coefficient expansion and independent
moment-cumulant formulas, then additivity, dilation and square-case checks.
Add small rectangular Haar-sum examples as numerical evidence. Monte Carlo
coefficient agreement supplements the algebraic oracle.
The empirical transform path is required: accept rectangular samples or squared
singular values with the same dimensions and zero-atom convention, and state
which ensemble quantity is estimated.

## B. New ensembles and expectation-polynomial connections

### E1: general-beta Hermite and Laguerre — first priority

Add compact tridiagonal/bidiagonal sampling for `beta > 0`, starting with
beta-Hermite and following with beta-Laguerre. The
[Dumitriu–Edelman matrix models](https://arxiv.org/abs/math-ph/0206043)
provide established constructions beyond the classical beta values 1, 2 and 4.
These models extend the spectral laws; their matrix-entry distributions differ
from the existing dense invariant samplers.

Agree on output representation before implementation: compact coefficients,
eigenvalues and an explicitly requested dense matrix have different storage
costs. Specify the joint-density convention, dimension, Laguerre shape domain,
scale and RNG ownership. Preserve existing dense samplers and their draw order.
For beta=4 comparisons, use one eigenvalue per quaternionic pair.

Acceptance requires Gaussian/chi-variable moment formulas, deterministic checks
with controlled draws, low-degree expectation-polynomial recurrences and seeded
law comparisons with classical models. Test shape-domain boundaries and
noninteger beta. Shared empirical-comparison code cannot be the sole oracle;
large distribution studies belong outside ordinary correctness tests.

### E2: Jacobi/MANOVA — second priority

Connect the existing Jacobi polynomial family to a real/complex MANOVA sampler.
Use nonsingular Gaussian factors and beta=1 or 2. The
[Edelman–Sutton model](https://math.mit.edu/~edelman/publications/beta-jacobi.pdf)
connects Jacobi laws, Haar decompositions and MANOVA.

Use a Hermitian representation or a generalized Hermitian eigenproblem rather
than assuming a nonsymmetric inverse product is Hermitian. State both factor
dimensions, the support `[0,1]`, the mapping of density exponents to Jacobi
parameters and the initial nonsingular domain. Rank-deficient endpoint atoms
are outside this sampler's scope.

Acceptance requires scalar beta-distribution moments, exact small-degree
coefficients/integrals and independently computed generalized eigenvalues.
Provide a matched sampler/expectation-polynomial example without identifying
expected-polynomial roots with mean eigenvalues.

## Initial implementation sequence

1. In [P00](packets/p00-baseline.md), reconcile merged work with the open issue
   list and review #37's current
   ownership. Inventory numerical limits and existing benchmark receipts;
   identify missing evidence without treating it as a failed feature.
2. In [P01](packets/p01-exact-moments.md) and
   [P02](packets/p02-inverse-finite-r.md), write the T1 mathematical/API
   proposal with independent fixed examples,
   including how incomplete cumulant sequences are handled. Implement only the agreed
   conversion/inversion scope in a focused PR.
3. In [P03](packets/p03-beta-hermite.md), define E1's density, normalization and
   compact output contract. Implement beta-Hermite first;
   [P04](packets/p04-beta-laguerre.md) follows after the first sampler's validation.
4. Deliver [P16](packets/p16-empirical-legacy-transforms.md)'s Cauchy, S, T and
   symmetric-S empirical routes through their individual issues, coordinating
   only the shared input/dimension conventions with P01.
5. Scope T2 in [P05](packets/p05-rectangular-convolution.md)/
   [P06](packets/p06-rectangular-cumulants.md) and E2 in
   [P07](packets/p07-jacobi-manova.md) independently. Keep their
   dimension/parameter dictionaries
   consistent with the accepted transform and ensemble conventions.

T1 and the first E1 sampler are the initial feature priorities. P05/P06 and P07
have their own scopes and prerequisites. Contributors can select one feature
without taking responsibility for a whole phase.

## Completion and review rules

A feature is ready for review when its mathematical definition, compatibility
effects, implementation, independent regression and executable example agree.
For transforms this also requires the empirical matrix counterpart, its declared
aggregation/estimand contract and an independently checked end-to-end example.
Follow [test contracts](https://github.com/jontb/FiniteFree/blob/develop/tests/README.md)
and run the relevant [development checks](development.md#checks). Runtime changes
need compiled/fallback coverage; release candidates also need installed-wheel,
minimum-dependency and artifact validation under the [release process](release.md).
Report exact commands, failures, skips and checks not run.

Review the roadmap after each milestone. Record each item's owner,
linked discussion/PR, evidence, remaining uncertainty and next decision. Track
completed validated workflows, reproduced/fixed defects, measured costs and
remaining limitations. Counts of new APIs, tests or plots alone do not establish
progress. Public mathematical claims require the supporting evidence.

New APIs and substantial changes require scope discussion under the
[contribution guide](https://github.com/jontb/FiniteFree/blob/develop/CONTRIBUTING.md).
Routine PRs target `develop`. Merge acceptance and release publication retain
their existing maintainer approval process.
