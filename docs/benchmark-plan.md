# Numerical benchmark preparation

This development protocol prepares three steps: measure prepared pencils,
measure prepared kernels and streaming summaries, then select one optimization
from the evidence. It changes no library API, mathematical convention or default.
The protocol identifier is **numerical-v1**. Smoke runs validate the tooling;
they do not establish throughput, a CPU/GPU crossover or a performance improvement.

## Fixed inputs and execution

Use a dedicated environment with the development dependencies from
[development setup](development.md#installation). The tools require no new runtime
dependencies. Run from the checkout with `PYTHONPATH=.`. Results must go into a
new directory; the runner refuses to overwrite one. Commit the tooling before
collecting a report intended for comparison. Dirty-tree smoke receipts record
the commit, changed paths and a hash of the runtime source and benchmark tools.

```bash
PYTHONPATH=. python scripts/run_numerical_benchmarks.py --profile smoke --output /tmp/finitefree-numerical-smoke
# Prepared baseline; launch separately when ready to collect measurements:
PYTHONPATH=. python scripts/run_numerical_benchmarks.py --profile baseline --output /tmp/finitefree-numerical-baseline
# On real CUDA hardware with compatible CuPy, require matched GPU cases too:
PYTHONPATH=. python scripts/run_numerical_benchmarks.py --profile baseline --gpu --output /tmp/finitefree-numerical-gpu
```

Each case runs in a fresh process with OpenBLAS/OMP/MKL thread counts fixed to 1.
The runner uses the active interpreter. One invocation has a **600-second total
budget** and **60-second per-case limit**, including reference computation; both
can be lowered. It stops on the first failed accuracy check, error, timeout or
interrupt and retains the manifest, earlier results and child logs. A partial
run stays incomplete. No failed or unrun case counts as a performance result.
Without `--gpu`, the manifest explicitly records CuPy as unrun. Requesting a GPU
requires working hardware; the runner never substitutes CPU measurements.

| Family | Smoke inputs | Baseline inputs | Repeated calls |
| --- | --- | --- | --- |
| Pencils | size 8, variables 2, batch 2 | size 16/64/128, variables 4, batch 1/16/128 | 2 smoke; 5 baseline |
| Hermite kernel | rank 4, points 4 | rank 8/32/64, points 16/64 | 2 smoke; 5 baseline |
| GUE summaries | degree 4, samples 8 | degree 8/32, samples 128/1024/8192 | 2 smoke; 5 baseline |

The seed is 862. Pencil coordinates/directions use an independent seed sequence
containing size, variable count and batch, so a case survives subset selection
and reordering unchanged. Kernel coordinates include a diagonal, adjacent floats
and separated points. Summary methods replay the same seeded GUE matrices,
generating one matrix at a time. Receipts include input digests; paired methods
with different digests invalidate the run. NumPy versions belong in comparisons:
a fixed seed alone does not promise identical draws across different versions.
The runner also rejects receipts from a changed source/tool snapshot.
Summary input digests capture the actual generator calls in the allocation pass
and must match the independent reference replay; timed passes exclude hashing.

## 1. Prepared pencils

The existing `benchmark_prepared_pencils.py` separates context initialization,
preparation, coefficient/input/output transfers, first calls, warm factorization,
reused-factor HVPs and cached gradients. Compare one matching input case per
backend/process. First calls are warm with respect to imports and device context;
the context initialization field is separate. Warm GPU timings synchronize the
current stream; CUDA event timings remain a distinct field.

An independent NumPy dense-solve reference checks sign/log-determinant and the
trace identities `g_j=tr(A^-1 A_j)` and
`(Hv)_j=-tr(A^-1 (sum_k v_k A_k) A^-1 A_j)`. It does not call prepared factors.
All outputs must be finite, signs must agree exactly and each scaled error
`max(abs(actual-reference)/(1+abs(reference)))` must be at most **5e-10**.
The existing CPU-prepared comparison and finite-difference HVP diagnostic remain
recorded; the finite difference is not the independent acceptance oracle.

Process RSS is cumulative. The pre-reference RSS field excludes later oracle
work but includes imports and earlier calls. Traced Python memory and the isolated
CuPy pool cover partial allocation scopes. **Total device peak memory remains
unmeasured without external telemetry.** Collect device telemetry separately,
record its sampling interval and distinguish process allocation from whole-device
usage. Make no total GPU-memory claim from these JSON fields alone.

## 2. Kernels and streaming

`benchmark_numerical_workflows.py kernel` compares prepared matrix evaluation
against nested scalar calls, producing the same full square kernel matrix.
Preparation and polynomial construction are separate; repeated coordinates reuse
the prepared basis metadata and scalar caches. Every matrix entry is compared
with SciPy's probabilists' Hermite basis divided by `sqrt(j!)`. Five selected
pairs also use exact rational SymPy Hermite sums. The scaled tolerance is
**2e-11**. This protocol covers the verified Hermite path; it does not characterize
generic bases or reused continuous quadrature/count laws.

The `summary` subcommand compares `StreamingEmpiricalComparison` against
`EmpiricalComparison` plus equivalent count/mean/std reductions. Both timings
include the same GUE generation and eigensolves. Counts use closed intervals
[-1,1] and [0,2]. Independent longdouble coefficient convolution and accumulation
check mean/std (scaled tolerance **2e-10**); count histograms must match exactly.
This checks numerical summaries of identical samples, not the sampling law or
statistical convergence. The reference shares the input sampler and eigensolver.

Memory uses a separate traced allocation pass, outside reported timing samples.
Summary allocation includes temporary bytes for hashing one matrix at a time.
`owned_ndarray_bytes` counts direct object array fields and excludes Python/FLINT
objects and returned copies. RSS includes imports and previous same-sized calls.
Streaming can use more time or memory on small inputs; retain that result. Report
storage against sample count before making a bounded-storage performance claim.

## 3. Optimization selection and acceptance

After a complete baseline, rank the recorded phase medians and allocation costs.
Profile the largest relevant phase in a separate invocation (for example with
`python -m cProfile -o /tmp/pencil.prof scripts/benchmark_prepared_pencils.py ...`).
Profiled timings are not comparable to the unprofiled baseline. Record the
selected workload, bottleneck, proposed change and expected tradeoff before
editing runtime code. The per-matrix LU dispatch is a candidate only if profiling
supports it; no implementation change is selected by this preparation.

Keep the baseline revision and raw results. Compare the candidate against that
revision on the same hardware, dependencies, thread limits and input digests.
Alternate baseline/candidate order across at least three fresh-process pairs.
Require all accuracy gates, a median improvement of at least 10% on the selected
phase, consistent improvement across the pairs, and no unexplained regression
over 5% in the other registered cases. Report dispersion, preparation/transfers
and memory tradeoffs as well as the best case. These are selection criteria for
this experiment, not ordinary correctness-test timing thresholds.

Before accepting a runtime change, run compiled and flag-fallback library/tooling
suites and the existing adversarial audit. Add an independent focused regression
if the change touches correctness or caching. If results are mixed or no candidate
clears the gates, retain the baseline and report that outcome. Publication and
release are separate actions.
