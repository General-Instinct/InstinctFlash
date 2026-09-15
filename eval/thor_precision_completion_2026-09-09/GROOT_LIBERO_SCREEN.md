# GR00T LIBERO-10: native versus FP8 on Thor

The fixed 20-scene screen completed with **native 20/20 (100%)** and **FP8
17/20 (85%)**: an observed difference of **−15 percentage points**. Three pairs
succeeded only with native precision; none succeeded only with FP8. This is a
small paired screen, not a population loss estimate, non-inferiority certificate
or proof of real-time performance. The default native path remains the fidelity
choice; this FP8 recipe cannot be described as loss-free.

| Discordant scene | Native | FP8 |
| --- | --- | --- |
| Task 6, seed 60000: white mug and chocolate pudding | Success, 210 steps | Unsuccessful at 720-step limit |
| Task 8, seed 80000: both moka pots on stove | Success, 504 steps | Unsuccessful at 720-step limit |
| Task 8, seed 80001: both moka pots on stove | Success, 402 steps | Unsuccessful at 720-step limit |

All outcomes are retained without retry, seed substitution or calibration changes
in response to evaluation. [Verified paired results](groot-libero-paired-comparison.json),
[native completion](groot-libero-native-complete.json),
[FP8 completion](groot-libero-fp8-complete.json).

## Matched protocol

Both arms use `nvidia/GR00T-N1.7-LIBERO` revision
`2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`, subdirectory `libero_10`, through
public Runtime on the same Thor boot. All seven inference files were verified
after transfer. The checkpoint's `libero_sim` embodiment, native processors,
four denoising steps, 16×7 decoded action chunks and eight-action execution
horizon are preserved. The paused simulator uses upstream GR00T `LiberoEnv.reset`
with no extra settling or replacement initial state; each episode has a 720-step
limit. Ten tasks use two fixed seeds each: task index × 10000 plus 0 or 1.
[Checkpoint verification](groot-libero-checkpoint-staged.json) /
[frozen scenes](groot-libero-campaign-scenes.json).

Native execution retains BF16 model computation. The explicit FP8 path uses
FP8 VLSA with native BF16 backbone/DiT computation; startup checks require four
packed E4M3 Q weights and an actual VLSA replay. Both arms use matmul TF32 off,
cuDNN TF32 on and cuDNN benchmark off. These compare whole implementations,
not isolated arithmetic. [Native identity](groot-libero-native-startup.json) /
[FP8 identity](groot-libero-fp8-startup.json).

The same real startup observation and prompt from task 0 / seed 9173 are used
in both arms, excluded from evaluation. Native float64 state bytes are retained
in the fixture and converted at the existing model boundary. Sequence-shape
VLSA calibration remains part of engine execution; this is not a claim that all
calibration is static. Imported LIBERO code, BDDL, initial-state assets and the
GR00T wrapper are bound to the declared checkout.
[Startup fixture](groot-libero-startup-preparation.json) /
[source binding](groot-libero-source-binding.json).

## Verification and replay

Verification covers all 40 request/result records, successful driver exits,
controller action shapes, finite values and digests, requested/resolved seeds,
frozen initial observations, bound endpoint identities and simulator source
fingerprints. Paired checkpoint, upstream, pipeline, package, server and startup
identities match. All 371 frozen source/config files and four compiled libraries
were rechecked unchanged after both campaigns.
[Final source recheck](groot-libero-source-final-recheck.json).

Raw requests, controller actions, logs, bound plans and verification scripts are
under `/home/ubuntu/ifl_eval/thor_precision_completion_20260909/groot-libero-campaign`.
The remote checkpoint, frozen `groot-libero-source-v3` overlay and server logs
are under `/home/guanming/ifl_eval/thor_precision_completion_20260909` on Thor.
The raw `run_groot_libero_server.sh` starts one precision at a time; each new
attempt requires a new receipt/output path. `bind_arm.py` binds a ready identity,
`run_arm.py` runs only the selected arm, and `verify_progress.py` validates its
retained results. With the raw tree restored, reproduce the comparison using:

```bash
PYTHONPATH=/home/ubuntu/InstinctFlash .venv-dev/bin/python \
  /home/ubuntu/ifl_eval/thor_precision_completion_20260909/groot-libero-campaign/compare_arms.py \
  --native /home/ubuntu/ifl_eval/thor_precision_completion_20260909/groot-libero-campaign/native-run \
  --fp8 /home/ubuntu/ifl_eval/thor_precision_completion_20260909/groot-libero-campaign/fp8-run \
  --output /tmp/new-groot-libero-comparison.json
```

Both policy servers and owned SSH tunnels were closed after verification.
These data concern this fine-tune and recipe, not the base DROID checkpoint,
other embodiments or other model families. The declared 100-pair minimum for
non-inferiority is not met. Longer-run latency, memory and energy are separate
measurements; simulation success does not certify them.
