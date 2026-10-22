# P16: Empirical counterparts for existing Cauchy, S and T transforms

**State:** planned. **Phase:** 1. **Roadmap:** required empirical support.

## Individual contribution issues

Each transform can be taken on independently, coordinating the small shared
input contract with [P01's issue](https://github.com/jontb/FiniteFree/issues/41):

- [#48: empirical Cauchy](https://github.com/jontb/FiniteFree/issues/48).
- [#49: empirical finite S](https://github.com/jontb/FiniteFree/issues/49).
- [#50: empirical finite T](https://github.com/jontb/FiniteFree/issues/50).
- [#51: empirical symmetric finite S](https://github.com/jontb/FiniteFree/issues/51).

One contributor is not expected to implement all four. Each issue includes its
own domain, independent references, empirical example and compatibility checks.

## Outcome

Supported empirical matrix/spectrum routes for the current Cauchy, S, T and
symmetric-S operations, preserving their finite-dimensional definitions.
The [P00 inventory](../p00-baseline-report.md#empirical-transform-inventory-and-assigned-gaps)
identified these gaps. This is required legacy support, separate from C01's
optional new transforms; P02 owns the empirical finite-R forward/inverse route.

## Prerequisites and decisions

[P01](p01-exact-moments.md)'s accepted matrix, effective dimension, pairing,
ownership and numerical conventions, plus an API scope discussion.

- Select a small supported adapter or entry points; do not require callers to
  build a characteristic polynomial themselves. Keep existing polynomial
  signatures, default branches and exact result types compatible.
- Declare per-matrix versus ensemble targets, zero atoms, poles, transform
  nodes, precision/range failures and matrix-level uncertainty. Averaging a
  nonlinear transform is a separate target from transforming mean coefficients.
- Keep ordinary positive-Hermitian S, PSD T and genuinely ±-symmetric S domains
  explicit. Quaternionic duplicate pairs do not imply ± symmetry. Unit-circle
  objects' existing rational-coefficient restrictions do not justify a general
  complex-unitary extension here.

## Work and touchpoints

1. Provide numerical per-sample Cauchy evaluation with the normalized resolvent
   identity, declared pole behavior and matched ambient zero embedding.
2. Provide finite S/T matrix or spectrum evaluation with the existing node and
   step conventions, strict positivity versus zero-root rules and explicit
   numerical coefficient/domain failures.
3. Provide symmetric-S support preserving `ratio` and `standard`, zero
   multiplicity and the positive-imaginary branch. Reject unsupported samples
   without silently enforcing spectral symmetry.
4. Add an end-to-end example for each route on actual sampled matrices, with
   independent fixed references and declared aggregation. Do not add a general
   sample-management or streaming framework.

Likely `finitefree/transforms.py`, the small P01 empirical adapter, public
exports, shared empirical contract tests and API/tutorial docs. Deliver through
the separate feature issues above, retaining each route as open until delivered.

## Independent acceptance

- Cauchy matches exact diagonal partial fractions and direct small-matrix
  resolvent traces, including zero embedding and poles.
- S/T match exact positive/semidefinite diagonal examples, rank/zero atoms and
  existing boundary regressions; include full-rank and rank-deficient Wishart
  comparisons with declared dimensions.
- Symmetric S matches exact spectrum `[-3,-1,1,3]`, both conventions and a
  Hermitian block linearization. Generic GOE/GUE draws cannot substitute for an
  exactly symmetric spectrum.
- Preserve the packet index's two-sample aggregation distinction, owned outputs,
  invalid-sample reporting and existing polynomial/precision behavior.

## Validation and handoff

Use focused empirical/transform/domain regressions in both backends, executable
examples and profile R. Record exact collected tests and results. Hand off an
updated P00 gap inventory; do not claim library-wide empirical transform parity
while this packet or P02's corresponding route is incomplete.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record).
