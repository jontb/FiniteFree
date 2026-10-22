# Implementation packets: Phases 1 and 2

This is the execution breakdown of the [Phase 1 and 2 feature roadmap](roadmap.md).
**P00 is ready for review; implementation packets remain planned.**
Each feature issue is one focused contribution, with explicit prerequisites,
scope, independent acceptance and validation. P16 groups four existing transforms
that are tracked in separately claimable issues. Proposed test files and commands in
packet cards become runnable when that packet is implemented. Split a packet if
its agreed scope no longer fits a focused review.

The planning baseline remains `develop` at `2e13bc2`. Live checks for this
breakdown found the same head and no open PRs. The
[P00 report](p00-baseline-report.md) reconciles the current issue snapshot and
records the baseline checks. Recheck ownership and linked work before starting.
Packet IDs are local planning identifiers, not GitHub issue numbers or assignments.

## Phase map

| Phase | Roadmap milestone | Packets | Completion gate |
| --- | --- | --- | --- |
| 0: establish the baseline | M0 | P00 | Pinned evidence and a scoped first feature |
| 1: transforms and beta ensembles | M1 | P01–P04, P16 | Algebraic and empirical transform paths, plus independently checked compact samplers |
| 2: rectangular and Jacobi models | M2 | P05–P07 | Matched matrix dimensions, algebraic/empirical rectangular transforms and a bounded-spectrum sampler |

These are priority groups, not a requirement to finish every earlier packet.
Use the prerequisites below. Keep each contribution focused on one feature;
independent paths allow parallel work by different contributors once shared
conventions are agreed.

## Claimable Phase 1 and 2 features

Each issue is a proposed feature labeled `enhancement` and `help wanted`.
Contributors can take on one sampler or transform; empirical support belongs
to that transform's acceptance criteria. Comment in the issue with a proposed
scope/API and check recent claims before starting. Public API choices still need
agreement; opening an issue does not mark the feature implemented.

| Phase | Feature | Issue | Coordination or prerequisite |
| --- | --- | --- | --- |
| 1 | Exact and empirical moments (P01) | [#41](https://github.com/jontb/FiniteFree/issues/41) | Settle moment/input conventions in the issue |
| 1 | Full inverse finite R (P02) | [#42](https://github.com/jontb/FiniteFree/issues/42) | P01 moment/dimension and empirical-input contract |
| 1 | Compact beta-Hermite sampler (P03) | [#43](https://github.com/jontb/FiniteFree/issues/43) | Independent of other new samplers/transforms |
| 1 | Compact beta-Laguerre sampler (P04) | [#44](https://github.com/jontb/FiniteFree/issues/44) | P03 compact-output/RNG contract |
| 1 | Empirical Cauchy (P16) | [#48](https://github.com/jontb/FiniteFree/issues/48) | Coordinate small input conventions with P01 |
| 1 | Empirical finite S (P16) | [#49](https://github.com/jontb/FiniteFree/issues/49) | Coordinate small input conventions with P01 |
| 1 | Empirical finite T (P16) | [#50](https://github.com/jontb/FiniteFree/issues/50) | Coordinate small input conventions with P01 |
| 1 | Empirical symmetric finite S (P16) | [#51](https://github.com/jontb/FiniteFree/issues/51) | Coordinate small input conventions with P01 |
| 2 | Rectangular convolution (P05) | [#45](https://github.com/jontb/FiniteFree/issues/45) | Independent of new scalar transforms/samplers |
| 2 | Rectangular cumulants/finite R (P06) | [#46](https://github.com/jontb/FiniteFree/issues/46) | P05 convolution and dimension contract |
| 2 | Real/complex Jacobi–MANOVA sampler (P07) | [#47](https://github.com/jontb/FiniteFree/issues/47) | Independent of beta-Laguerre and other Phase 2 features |

These issues were opened unassigned. Check their live state and discussions;
this table does not reserve them or require one contributor to finish a phase.

## Core packet queue

| ID | Deliverable | Hard prerequisites | State |
| --- | --- | --- | --- |
| [P00](packets/p00-baseline.md) | Baseline evidence and scope decisions | None | Ready for review |
| [P01](packets/p01-exact-moments.md) | Exact root moments and empirical matrix moments | P00 | Planned |
| [P02](packets/p02-inverse-finite-r.md) | Full inverse finite R for algebraic and empirical inputs | P01 | Planned |
| [P03](packets/p03-beta-hermite.md) | Compact beta-Hermite sampler | P00 | Planned |
| [P04](packets/p04-beta-laguerre.md) | Compact beta-Laguerre sampler | P03 | Planned |
| [P05](packets/p05-rectangular-convolution.md) | Rectangular coefficient convolution | P00 | Planned |
| [P06](packets/p06-rectangular-cumulants.md) | Algebraic and empirical rectangular cumulants/finite R | P05 | Planned |
| [P07](packets/p07-jacobi-manova.md) | Real/complex Jacobi–MANOVA workflow | P00 | Planned |
| [P16](packets/p16-empirical-legacy-transforms.md) | Empirical counterparts for existing Cauchy, S, T and symmetric S | P01 and an agreed adapter scope | Planned |

The first feature priorities are **P00 → P01 → P02 → P03**. P03 is independent
of P01/P02 mathematically and can move earlier. P04 follows P03's compact/RNG
contract; P05/P06 and P07 have their own prerequisite paths. No contributor is
expected to complete an entire phase.
P00 selected P01 and prepared its [scope proposal](p01-scope.md). P16 was added
from the audit to close existing empirical support gaps without expanding P01;
the library-wide counterpart requirement remains open until those routes and
P02's empirical finite-R route are delivered.

## Dependency paths

| Path | Order | Reason |
| --- | --- | --- |
| Exact transforms | P00 → P01 → P02 | Establish moment/dimension conventions before exposing reconstruction |
| Existing empirical routes | P00 → P01 → P16 | Reuse small matrix-input conventions while preserving each legacy transform's domain |
| Beta ensembles | P00 → P03 → P04 | Reuse the accepted compact-output and RNG conventions |
| Rectangular algebra | P00 → P05 → P06 | A transform must linearize the implemented, independently checked convolution |
| Bounded spectra | P00 → P07 | Existing Gaussian/Wishart tools suffice; P04 is not a mathematical dependency |

## Required empirical matrix counterparts

**Every transform must support its empirical random-matrix counterpart.** This
is a required part of the public feature contract for existing and new transforms,
including inverse operations. P00 inventories legacy gaps. Each affected
transform packet implements its paired empirical route and
an end-to-end example; optional Monte Carlo research is additional evidence.

1. **Usable inputs:** provide a supported direct matrix/spectrum input or a small
   reusable adapter. State Hermitian/positive/unitary or rectangular domains,
   compact-model conversion, paired eigenvalues, dimensions, zero atoms and
   ownership. Preserve existing retained/streaming comparison defaults.
2. **The same conventions:** match finite dimension, aspect ratio, scaling,
   moment/cumulant normalization, transform nodes, branches and truncation.
   Keep finite transforms distinct from similarly named limiting/free transforms.
3. **A declared statistical target:** specify per-sample transforms, a transform
   of estimated expected coefficients, or a transform of an estimated mean
   spectral measure. Averaging and a nonlinear transform generally cannot be
   interchanged. Inverse operations must support the corresponding empirical
   estimates without implying recovery of a unique matrix law.
4. **Visible numerical/statistical limits:** state precision and conditioning,
   how matrix-level uncertainty is reported, and handling of invalid samples,
   poles, missing moments or domain violations. Do not silently filter samples
   or turn floating observations into certificates through exact conversion.
   Averaged empirical coefficients can fall outside a transform's root domain;
   report that status explicitly without automatic positivity/root repair.
5. **Independent acceptance:** include exact diagonal/small-matrix identities
   and a seeded ensemble example using the actual sampled matrices. Retained
   and streaming empirical paths must estimate the same declared target when
   both are offered. Two routes sharing an implementation are insufficient alone.

For an aggregation regression, use the two diagonal samples with spectra `[1,3]`
and `[3,5]`. Their mean second spectral moment is 11. Their mean characteristic
polynomial is `(x-3)**2`, whose root second moment is 9. In the current degree-two
finite-R convention, their individual second cumulants are both 2, whereas the
second cumulant of the mean polynomial is 0. The empirical interface must label
which of these quantities it returns.

This requirement is part of each transform packet's completion gate, rather than
a later optional benchmark or example-only follow-up. If supporting it makes a
packet too large, split its internal work while keeping algebraic and empirical
support in the same feature milestone.

## Starting a packet

1. Read its card, [AGENTS.md](https://github.com/jontb/FiniteFree/blob/develop/AGENTS.md),
   the [code map](code-map.md), [API contracts](api.md) and relevant test contracts.
2. Refresh current `develop`, issues, discussions and PRs; preserve unrelated
   local work. Confirm prerequisite evidence against that source revision.
3. Record the scope discussion for a new API, convention or substantial change,
   following [CONTRIBUTING.md](https://github.com/jontb/FiniteFree/blob/develop/CONTRIBUTING.md).
   Settle the card's open choices before runtime implementation. A planning card
   is not an accepted signature or a release approval.
   For a transform, settle its empirical input, normalization, aggregation order
   and uncertainty contract in the same discussion.
4. Create a focused branch from current `develop`. Implement only the selected
   packet, keeping code, its independent regression, an example and relevant
   documentation together. Existing public signatures and explicit options stay
   compatible unless the accepted scope says otherwise.
5. Run focused checks while iterating and the appropriate final profile below.
   Proposed module names are starting points; if changed, update the card and
   verify the intended tests were collected. A missing test is not a pass.

If a source definition or oracle contradicts the proposed scope, record the
discrepancy and revise that packet instead of weakening its acceptance criteria.

## Shared validation profiles

Commands run from the repository root in an environment prepared using the
[development guide](development.md#installation). Packet-specific commands are
additional focused checks. These profiles describe future packet validation;
their appearance here does not report them as executed.

### D: documentation

Run the synchronizer when README changes. Generate the interactive assets before
a strict build; keep their ignored output out of the diff.

```bash
python scripts/sync_docs.py
PYTHONPATH=. python scripts/check_readme_examples.py
PYTHONPATH=. python examples/interactive_geometry.py --output docs/visuals/interactive
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python examples/spectral_dashboards.py --output docs/visuals/interactive
python -m mkdocs build --strict
git diff --check
```

### R: runtime and tooling

The full runner builds the source extension, runs compiled and flag-fallback
suites, checks the fixed adversarial controls, executes examples and builds docs.
It stops on the first failure; name any later checks that were not reached.

```bash
python scripts/check.py --full
git diff --check
```

New numerical contracts need independent expected values as well as parity or
round-trip checks. Keep existing adversarial cases, ownership, precision and RNG
coverage. Use fixed deterministic oracles in ordinary tests; large distribution
studies and performance comparisons require a separate bounded protocol.

### E: experiments

First run the experiment tool's focused regression under R if tooling changes.
Then execute only the accepted case list and budget from its packet. Preserve
source/input hashes, seeds, dependency versions, hardware, thread limits,
completed/failed/unrun cases and raw results. Store local receipts under a fresh
`experiments/packets/<packet-id>/<run-id>/` directory or another agreed durable
location; `experiments/` is ignored. Put a concise evidence summary and receipt
hashes in the tracked report/PR so ignored files are not the only handoff.

A timeout or failed reference leaves a run incomplete. A completed valid study
may support a negative conclusion. Neither outcome establishes an unmeasured
performance, sampling or asymptotic claim. Agree compute scope before launching
expensive or optional sweeps; never silently enlarge it after a timeout.

### I: integrated candidate

Follow the existing [release process](release.md) when preparing a candidate
containing selected features. Validate the exact integrated commit, build and
inspect artifacts, install its wheel outside the source tree, and check minimum
dependencies and the supported platform matrix. A local Linux build alone does
not validate the production wheel matrix. Publishing remains a separate action.

## Packet handoff record

Update the selected card or its linked PR with:

- ID, state, owner, branch/commit and scope-discussion link.
- Accepted choices, satisfied prerequisites and actual changed paths.
- For transforms, the paired empirical route, statistical target, normalization
  and independently checked exact-versus-matrix example.
- Independent oracle and exact validation commands, with counts, failures,
  skips and checks not run.
- Receipt/report locations and hashes for measurements; retained uncertainty,
  compatibility effects and material agent assistance.
- One next decision: ready for review, narrow follow-up, incomplete validation
  or documented deferral. Tests passing does not grant merge/release authority.

Use planned → scoped → implementing → validating → ready for review → accepted
for implementations. Research cards may finish as a documented no-go or negative
result after valid evaluation. Missing evidence remains incomplete. Recheck and
update this queue at each milestone; avoid changing packet IDs once work is linked.
