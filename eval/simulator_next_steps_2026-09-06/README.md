# Repeatability, native checkpoint onboarding, and edge latency

Experiment root: `/home/ubuntu/ifl_eval/next_steps_20260906`.
This extends the [five-family screening](../simulator_quality_2026-09-06/README.md).
No distillation, reduced NFE, quantization or replacement attention implementation
was introduced in these experiments. Numerical equivalence, closed-loop quality
and device latency remain separate claims.

## V2 attribution

A is the native original; B is the default in-process Runtime. All three comparisons
use the same frozen RoboTwin scenes, `fresh-clutter-bounds-v1`, serial scheduling,
`adjust_bottle` and `beat_block_hammer`, one seed per task and setting. Clean and
randomized are separate. There are 24 completed episodes across A/A, B/B and A/B.

| Comparison | Setting | Success, first → second | Identical executed traces |
| --- | --- | --- | --- |
| A/A | clean | 2/2 → 2/2 | 0/2 |
| A/A | randomized | 2/2 → 2/2 | 0/2 |
| B/B | clean | 2/2 → 2/2 | 0/2 |
| B/B | randomized | 2/2 → 2/2 | 0/2 |
| A/B | clean | 2/2 → 2/2 | 0/2 |
| A/B | randomized | 2/2 → 2/2 | 0/2 |

Each setting has only two paired episodes: the descriptive central 95% paired
score interval for the zero success delta is approximately ±65.8 percentage points.
These are attribution probes, not new accuracy certification or coverage expansion.
Tasks were chosen before running this triad; the hammer task appeared in an earlier
discordant pilot. Results cannot estimate performance across all RoboTwin tasks.

The A/B original rollouts supplied 12 real policy inputs, including camera images,
state, prompt and **actual initial noise tensors**. Each endpoint replayed all 12
inputs three times, with no simulator feedback. Noise was identical in every replay.
The comparison with the first repetition excludes that repetition's trivial
self-comparison:

| Endpoint / comparison | Identical action chunks | Largest absolute action-value difference |
| --- | --- | --- |
| Original repetitions 2/3 versus repetition 1 | 13/24 | 0.0114672184 |
| Runtime repetitions 2/3 versus repetition 1 | 6/24 | 0.0101678474 |
| Original versus Runtime at corresponding input/repetition | 15/36 | 0.0131003596 |

Even the original is not repeatable at the action-byte level under this execution
setup. In one A/A first call, identical observations and identical noise produced
different actions before any simulator feedback. Runtime also varies internally.
This rules out attributing all A/B differences to simulator feedback or to Runtime
alone. It does **not** identify a particular nondeterministic kernel, establish that
Runtime adds no error, or convert V2's NUMERIC classification to BITEXACT.
Maximum action-value differences mix native joint/gripper dimensions and are not
percentages of task accuracy loss. Success ties do not establish zero quality loss.

The endpoint processes stay alive across the triad and replays. Startup uses a
shape probe, not exhaustive warmup of every task shape. A/A, B/B and A/B run in
that order; this is not a counterbalanced study across fresh processes or a
per-kernel ablation. The frozen H100 V2 receipt predates explicit numerical-flag
fields: its model/source/packages and execution settings are recorded, but flags
must not be retroactively inferred as measurements. The current server now records
TF32, cuDNN benchmark and deterministic-algorithm settings for future runs.
Recording actual noise adds synchronization/copy overhead to both arms; none of
these shared H100 simulator timings are used for speed claims.

[Machine-readable analysis](attribution-and-checkpoint.json) includes first-call
comparisons, raw-evidence hashes, paired intervals and all cross-arm replay rows.
Reproduce it using the installed package with `eval` and `serve` extras:

```sh
python analyze.py --root /path/to/next_steps_20260906 --output analysis.json
```

## A genuinely different native checkpoint

The standard CLI completed the entire checkpoint-view → plan → prepare scenes →
doctor → run → report → export → verify workflow for
`nvidia/GR00T-N1.7-LIBERO@2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21/libero_spatial`.
It uses 10 tasks × two seeds, giving 40 episodes across the two arms.

| Native fine-tune | Original success | Runtime success | Identical executed traces |
| --- | --- | --- | --- |
| LIBERO-Spatial | 20/20 | 20/20 | 20/20 |

The paired success delta is 0 percentage points, with descriptive central 95%
interval approximately [-16.1, +16.1]. This is screening, not a release gate.
All 20 observed controller traces agree; hidden states and untested inputs remain
outside that statement.

This is a different weight subdirectory under the **same pinned Hub revision**,
not a different revision or a renamed copy of LIBERO-10:

- Previous LIBERO-10 inventory: `1ae2e4c6a0b81336db8bfac0c559e4504108c85f0c6f58b6525efa9484e81453`.
- LIBERO-Spatial inventory: `1b7d294c5740df4e14171caee8bca689ffc2e23e30f347fdc18c516b2998c174`.

The native reset, image transforms, EEF observation, gripper conversion and
8-action execution schedule are preserved. The adapter also implements the
matching Object and Goal subdirectory contracts; those experiments have not run.
A new embodiment or action space still requires a reviewed native adapter. This
validates reusable onboarding within a contract, not arbitrary-checkpoint universal
compatibility. Scores from different fine-tunes/task suites are not a leaderboard.

## Integrated workflow

The [simulator guide](../../benchmarks/vla/SIMULATOR_EVALUATION.md) documents
`checkpoint-view`, `replay-policy`, `measure-policy`, `realtime-report` and the
campaign lifecycle. Plans pin checkpoint subdirectories; coverage retains them;
wrong-suite substitutions are refused. Failed recording attempts are archived with
hashes before retry. The final installed package is verified independently of the
source working directory, and the six previous evidence bundles remain valid.

## Native Jetson Thor measurements

Both arms ran on an NVIDIA Jetson AGX Thor Developer Kit, in separate processes,
with the same pinned V2 weights, native code and Qwen configuration/tokenizer.
Three real inputs from the clean `adjust_bottle` episode (seed 50100) were cycled
with their actual recorded noise. Each arm has eight warmup calls and 128 measured
calls, with NFE=10 and a 50-action chunk. An additional first prediction forces
model initialization before warmup.

| Native path | p50 | p95 | p99 | Maximum |
| --- | --- | --- | --- | --- |
| Original | 717.83 ms | 735.92 ms | 744.10 ms | 787.24 ms |
| Runtime, explicit `in_process` | 735.54 ms | 748.46 ms | 760.65 ms | 768.63 ms |

The Runtime explanation records that the Thor planner **declined graph capture**;
its log confirms the static-KV graph backend was not installed and the upstream
path ran. Explicit `in_process` also prevented FP8 engine placement. This experiment
shows no speedup for that native Runtime path. It is a different operating path
from the historical README's Thor FP8 engine rows. A plan saying BITEXACT is not an
end-to-end equality certificate; the V2 probe conservatively remains NUMERIC and
makes no new action-equivalence claim on Thor.

Measurements include CPU observation preprocessing through synchronized ready CPU
actions, including diagnostic noise-copy overhead. Camera acquisition, network,
actuators and sustained controller scheduling are excluded. No power/clock settings
were changed; telemetry is retained. The three input cases and one sequential
process run per arm do not establish a general latency distribution or a statistically
proven wrapper overhead. Model initialization plus first prediction took approximately
57/55 seconds; those values exclude the preceding checkpoint-inventory validation
and are not cold-cache startup comparisons.

| Explicit illustrative budget | Call deadline | Original misses | Runtime misses |
| --- | --- | --- | --- |
| 20 Hz, execute 50 actions, inference overlaps execution | 2500 ms | 0/128 | 0/128 |
| 50 Hz, execute 50 actions, inference overlaps execution | 1000 ms | 0/128 | 0/128 |
| 20 Hz, blocking call each control period | 50 ms | 128/128 | 128/128 |
| 50 Hz, blocking call each control period | 20 ms | 128/128 | 128/128 |

For the two overlapping-execution scenarios, the original already meets the observed
budget: no lossy optimization is needed to satisfy those assumptions. This does not
mean the policy reacts to fresh observations at 20/50 Hz. The user's actual chunk
execution, overlap and reaction deadline have not been specified. Deployment remains
uncertified; failing a blocking scenario also does not automatically authorize distillation.

Both arms recorded matmul TF32=true, cuDNN TF32=true and cuDNN benchmark=false.
These are the native settings in this comparison, not a newly enabled candidate
precision change. Forward dependencies include Torch 2.11.0+cu130, Transformers
4.57.3, NumPy 2.2.6 and FlashAttention2 2.8.4. The H100/Thor software stacks differ;
this is a within-Thor latency comparison, not cross-device bit-exact evidence.

The isolated installation compiled the [official FlashAttention2 source](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/setup.py) for SM110.
Only its backward-build feature was disabled; forward kernel sources were unchanged.
Triton's bundled assembler rejected `sm_110a`, so both successful arms explicitly
used the installed CUDA 13.2 `ptxas` via `TRITON_PTXAS_PATH` and
`TRITON_PTXAS_BLACKWELL_PATH`. The resulting native MoE path ran successfully;
no SDPA shim or alternate MoE algorithm was substituted. Failed startup attempts,
build source revision/patch, binary hashes and telemetry remain in the evidence.
`qwen-vl-utils==0.0.11` was added after the original run to satisfy the Runtime's
import-availability check; the pinned native forward source does not reference it.

[Budget calculations](thor-budget-report.json) retain every scenario and measurement
file hash. Reproduce with the installed package:

```sh
python assess_thor.py --measurements thor \
  --budgets control-budgets --output local-thor-report.json
```

## Verification and provenance

[Evidence index](evidence-index.json) records the four new simulation bundle hashes,
the two fixed-input replay files, the Thor archive, frozen execution revisions,
installed wheels and final source archive. The six previous simulation bundles and
all four new ones pass verification with the final installed package. Eight real
budget assessments also ran through `instinctflash eval realtime-report`.
Targeted tests cover input/noise replay, checkpoint variants, budget handling,
reporting, paired execution and retention of failed recording attempts. The relevant
32-check suite, 21-check reporting/runner suite, 16 pipeline checks, updated
11-check resume/coverage suite and 27-check installed-environment suite passed;
these groups overlap and are not a count of distinct tests. README local links and
`git diff --check` also pass.

The final evaluation pipeline SHA-256 is
`182ca51cd67a9b3217d0c698548dff4c17e407e2df4aaa228a5980cceb59b006`.
Execution snapshots were frozen independently; later documentation and robustness
changes did not alter running experiments. Failed setup attempts and earlier source
archives were preserved. All four local model servers, both Thor model processes
and this experiment's telemetry process have exited; local GPUs 4–7 are released.
