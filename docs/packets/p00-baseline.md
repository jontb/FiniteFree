# P00: Baseline evidence and first scope decisions

**State:** ready for review. **Phase:** 0. **Roadmap:** M0.

The [baseline report](../p00-baseline-report.md) records the pinned audit,
contributor checks, empirical support gaps and remaining validation limits.
The selected next feature has a [concrete P01 scope proposal](../p01-scope.md);
public API agreement is still required before runtime implementation.

## Outcome

A pinned evidence inventory and an agreed scope for the first feature. This packet establishes what is already present and which checks remain open.

## Prerequisites and decisions

None. Start from current develop and preserve any unrelated local work.

- Record the source revision, runtime/tool versions, optional backends and intended first packet. Recheck issue discussions, assignments, recent PRs and existing local receipts.
- Classify #35/#36 as implemented if their merged source is still present; check #37's ownership before offering it as side work. Changing GitHub state is a separate authorized action.
- Choose the first transform or sampler scope and record its discussion link. For each transform, include the required empirical matrix input and declared statistical target. Do not settle every future API in this packet.

## Work and touchpoints

1. Review finitefree/__init__.py, transforms.py, ensembles.py, numerical.py, the code map, API and test contracts. Record existing exports, normalization, RNG and precision behavior relevant to the selected feature.
2. Run the existing contributor baseline in a prepared environment. Preserve exact logs; distinguish a reproduced baseline defect from missing setup or an unrun backend.
3. Record prior fixes and cache behavior as existing capabilities. Reproduce any reported optional-CuPy typing issue in its stated environment before classifying it as an active defect.
4. Inventory the empirical counterpart of each existing Cauchy/R/S/T/symmetric-S operation: supported matrix/spectrum input, dimension/scaling, aggregation order, domain and comparison example. Record gaps explicitly and assign their resolution to the affected transform scope.

Existing contracts in docs/api.md and tests/README.md; one concise baseline/decision record linked from this card or the first PR. Runtime source changes are outside this packet.

## Independent acceptance

- The inventory identifies the exact source and links existing implementations, tests and current issue/PR evidence.
- Every reported pass has a log; skips, baseline failures and checks not reached have an explicit disposition. No unexplained failure affecting the chosen work path is dismissed.
- The first feature has a bounded scope and independent expected examples. Unrelated documentation chores do not block its start.
- A transform feature scope includes its empirical counterpart and an explicit per-sample or ensemble target; a coefficient-only scope does not satisfy the revised feature requirement.

## Validation

Use profile R from the packet index for the baseline; use D for documentation changes. A missing dependency or optional GPU is recorded as missing/unrun, not successful validation.

## Handoff and limits

Hand off the evidence summary, prerequisite decisions and the next selected packet. No new sampler, transform, benchmark sweep, issue closure or release action.

Use the [packet handoff record](../implementation-packets.md#packet-handoff-record) and the shared validation profiles. See the [roadmap](../roadmap.md) for the mathematical motivation.
