# Cosmos distillation → Thor deployment study (active)

Objective: continue distillation for Cosmos Edge/Nano, combine trained students
with Flash inference acceleration, and verify real-time execution with task
quality evidence. This study is not complete and does not establish real-time
closed-loop performance.

Cross-repository handoff is at
`/home/ubuntu/InstinctCompress/coordination/FLASH_COSMOS_REALTIME.md`.
It requests the Compress agent's current checkpoint/training ownership, next
few-step plan, Nano coverage and evaluation contract. The existing Compress agent acknowledged the official queued handoff on
2026-09-12 at 06:38 UTC; its reply is in `COMPRESS_COSMOS_REALTIME_REPLY.md`. Flash owns these Thor measurements and has not changed Compress
training code or occupied its H100s.

## Current deployment candidates

- V5 selected Edge RCM EMA update128: actual native merged export, fixed SDE
  `[1,.75,.5,.25,0]`, literal CFG4, zero action padding. Prior Compress export
  audits include native → Runtime parity. New Thor performance still needs its
  own qualification; those audits do not certify task success.
- Historical Edge DMD2 `dmd2_120_v2`: genuinely trained one-step native export,
  SDE `[1,0]`, CFG3. This was a small integration study: its historical teacher
  and recorded-command error did not improve over the untrained one-step
  control. It is a latency/compatibility candidate, not a quality winner.
- Latest V15: complete four-step CFG4 training and native cold-reload evidence,
  but no clear quality superiority to UniPC4 or fixed-last training. Its latest
  native export lacks a Flash execution declaration. Do not infer a complete
  deployment contract or relabel intermediate training exits as few-step models.
- Nano: no deployed trained student located yet; coordination/training remains
  outstanding. Edge findings do not qualify Nano.

The two exported candidates are copied, with checksum comparison, into isolated
Thor directories under `/home/guanming/ifl_eval/cosmos_distill_20260912`.
No original checkpoint is modified. `verify_artifact` checks full package
integrity before loading. `benchmark_student.py` keeps the checkpoint's complete
sampler and observes its actual evaluation count. It measures six warmup and ten
steady-state requests, full finite 32×8 outputs, two prompts and resets.

Each candidate has native and instance-local cuDNN BF16 attention arms with
existing checked pointwise/layer graph optimizations. These are explicit
**qualification experiments**: production cuDNN admission currently remains
restricted to the previously screened official CFG3/four-step configuration.
No student admission guard is silently broadened. No conditioning cache is
claimed for these new sampler/CFG combinations.

The fixture is the existing Flash latency fixture, not a new distillation
training dataset or a closed-loop task-quality evaluation. Student-to-student
latency differences confound weights, padding, guidance and schedule; only each
student's native/cuDNN pair isolates attention implementation.

## Reproduce

With Flash and the frozen Compress plugin on PYTHONPATH in the Thor Cosmos venv:

```bash
python run_pair.py /path/to/student /new/output \
  --fixture /path/to/va_eval_obs.npz
```

The runner takes `/tmp/thor_gpu.lock`, serializes fresh processes, and refuses
existing logs/results. The script validates package integrity, full sampler
counts, finite actions and actual cuDNN execution, and records source hashes.

## Real-time acceptance still outstanding

The user was asked for the intended controller deadline. Until that is fixed,
report latency and action-coverage scenarios separately. DROID dataset actions
are sampled at 15 Hz; the conditioning FPS field alone does not establish a
controller's executed chunk. Execution of 8, 16 or 32 such actions would cover
0.533, 1.067 or 2.133 seconds, respectively. Those are conditional arithmetic
budgets, not proof of a deployed control loop meeting its deadline. Fresh-input
replanning, p95 latency, deadline misses and task success must be verified.

## Completed first Thor deployment screen

| Student | Native p50 ms | cuDNN p50 ms | cuDNN p95 ms | Attention speedup | Max action delta |
|:--|--:|--:|--:|--:|--:|
| v5 | 2555.1 | 1469.7 | 1473.4 | 1.74× | 0.181519 |
| one-step | 711.8 | 442.2 | 443.3 | 1.61× | 0.078125 |

All four processes completed successfully, each with 16 full finite action
chunks. Actual callback counts are 64 for each V5 arm and 16 for each one-step
arm. Paired manifests, sampler contracts, fixture and shared source hashes
match; each numerical arm executed cuDNN. See [receipts](receipts/).
No task-quality or full-goal completion is claimed. No GPU job from this first
paired study remains running. Production admission has not been broadened.

## Coordinated continuation

Compress owns new genuine one/two-step and guidance-distilled training, including
Nano, with separate new-study controls. Flash owns Thor execution qualification.
See [Nano latency budgets](nano-budget/README.md) and the [current V14/V15
deployment packages](deployment/README.md). Real-time acceptance and trained
student quality remain outstanding.
