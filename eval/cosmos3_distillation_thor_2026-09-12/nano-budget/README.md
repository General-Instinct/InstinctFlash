# Nano step/guidance latency budgets on Thor

These runs use **original Nano weights**, native UniPC, BF16 cuDNN attention and
explicitly changed step/guidance settings. They are not trained students, quality
results, or a claim of the lowest achievable latency.

| Operating point | p50 ms | p95 ms | Actual branches per request |
|:--|--:|--:|--:|
| One step, CFG3 | 1496.9 | 1556.9 | 2 |
| One step, CFG1 | 801.9 | 805.0 | 1 |
| Two steps, CFG1 | 1562.5 | 1567.4 | 2 |

Each fresh process recorded six warmup and ten measured requests, finite full
32×8 actions, actual velocity branch counts and real cuDNN execution. Source,
checkpoint index/revision, fixture and benchmark hashes match across arms.
[Raw receipts and summary](receipts/summary.json).

The source checkpoint still declares its original schedule. The experiment
passes an explicit action-NFE override and replaces the service's immutable
configuration with a new guidance value before prediction. Both changes are
recorded as OPERATING-POINT overrides; they are not silently presented as
checkpoint defaults. Actual branches are counted outside the per-layer graphs.
Production attention/cache admission is not broadened.

At a hypothetical 15 Hz action frequency, executing 16 actions before replanning
covers 1067 ms: the one-step/CFG1 arm had 0/10 measured misses against that budget;
the other arms had 10/10. For 8 actions (533 ms), all three had 10/10 misses. These
are conditional sample counts, not an inferred controller contract or verified
closed-loop realtime behavior. Fresh-input age, safety margins and quality
remain unqualified.

The initial script tried assigning the frozen service configuration's guidance
field and failed before action measurement. The successful version replaces the
configuration using `dataclasses.replace`; the failed attempt remains under
`/home/guanming/ifl_eval/cosmos_distill_20260912/nano-budget-results` and is not
included in the table. Successful runs are under `nano-budget-results-v2`.

The next training comparison should explicitly examine single-branch guidance
distillation: it changes this Nano workload's measured budget materially. A
trained student's exact SDE grid and runtime need their own fresh measurements.
