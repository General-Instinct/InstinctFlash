# Thor precision comparison: current API and historical controls

Latest matched speed sweep: [2026-09-10 Thor native/FP8 results](../thor_fp8_2026-09-10/README.md), covering all nine README configurations. The measurements below retain their original scopes and quality evidence.

FP8 improves latency for some measured models, but currently regresses on Cosmos.
The gain depends on the model, implementation and baseline. Compare it
against accelerated native execution when deciding whether to trade numerical
fidelity for speed. Engine implementation and precision changes are separate
contributors. None of the current API measurements below certifies task quality.

## Decision table

All times below are Thor median action-chunk generation latency in milliseconds.
Compare within a row: checkpoints, chunk sizes and workloads differ across rows.
Native precision does not itself establish BITEXACT equivalence; exact retained
actions establish equality only for the tested inputs.

| Model / workload | Native eager | Accelerated native precision | FP8 | FP8 gain against measured native alternative |
| --- | ---: | ---: | ---: | --- |
| pi05 BASE, 50 actions, fixed state | — | 1218.24 (FP32) | 76.11 | 16.01×; 16 warm chunks, different native dtype from LIBERO |
| pi05 LIBERO v044, 50 actions, fixed state | — | 339.05 | 61.26 | 5.54×; eight warm chunks only |
| VLA 4B, 25 actions | 704.40 | 355.00 | 217.97 | 1.63× |
| VLA V2, 50 actions | 764.29 | 423.50 | 238.98 | 1.77×; prompt-reset latency exceeds 1 s |
| GR00T DROID, 40 actions, cached weight casts | 110.30 | Capture slower in separate pair | 127.40 | 15.5% higher latency |
| Cosmos Edge, 32 actions, scale-optimized study | 3454.31 | Not qualified | 3561.41 | 3.10% higher latency |
| Cosmos Nano, 32 actions, scale-optimized study | 10321.11 | Not qualified | 10392.84 | 0.70% higher latency |
| DreamZero | Diagnostic controls only | Not qualified | Diagnostic controls only | No demonstrated gain |
| LingBot VA LIBERO-long, full 20V/50A | 4160.92 | Not qualified | 2078.53 | 2.00×; early-history short test |
| LingBot VA RoboTwin, full 25V/50A | 5564.59 | Not qualified | 2815.17 | 1.98×; early-history short test |

Cosmos rows use the measured scale-reduction candidate subsequently incorporated
into the local production module; the preserved receipts identify its measured source.

These are whole-runtime gains, not the isolated contribution of FP8 arithmetic.
The historical pi05 FP16-engine versus FP8-engine control below gives the closest
available incremental comparison: 79.97 → 44.89 ms, or 1.78×.

## Current public Runtime: VLA 4B

Same pinned RoboTwin checkpoint, native camera/state/action processors, live vision
on every call, ten denoise steps, 25×14 decoded replies. Four warmup and 32 measured
calls per process, recorded images and synthetic states. One process per arm;
these are short diagnostic measurements, not sustained tail-latency qualification.

| Execution | p50 ms | p99 ms | Setup seconds |
| --- | ---: | ---: | ---: |
| Native BF16 eager | 704.40 | 713.59 | 44.06 |
| Default native BF16 with capture | 355.00 | 362.64 | 43.90 |
| FP8 language/action, native BF16 vision | 217.97 | 221.77 | 18.50 |

The current FP8 route is **3.23×** faster than eager and **1.63×** faster than
captured native execution (38.6% lower median latency). Native capture alone is
1.98× faster. The previous FP8 run measured 213.53 ms; the latest rerun corrects
the graph receipt and records 36 language/action graph replays. Vision stays eager.

All 36 native eager/capture action arrays are exactly equal. Native capture versus
FP8 has max absolute decoded-action difference 0.30121 and mean absolute difference
0.02107 across these samples. These mix action dimensions and units, and **are not
success-rate loss**. Seeds match per call; this is not an isolated FP8 arithmetic
control: the native constructor enables matmul TF32, the engine disables it, and
their implementations differ. Both disable cuDNN benchmark.

The qualified native source uses eager vision attention and SDPA expert attention
on Thor. It is not an unmodified upstream FA2 installation. Source paths, hashes,
recipes and all arm receipts are in [the comparison JSON](vla4-public-comparison.json).

The historical ~99 ms engine used FP16 vision. Real native-processed camera inputs
exposed vision overflow, followed by finite but image-insensitive actions. That
speed is not a valid deployment result. The corrected route retains BF16 vision
and its current paired simulator screen is reported below.

## Current VLA 4B closed-loop screen

All 40 paired RoboTwin scenes completed on Thor. Clean measured native **15/20**
and FP8 **14/20** (−5 percentage points); randomized measured **15/20** and
**16/20** (+5 points). Clean has two native-only and one FP8-only successes;
randomized has three and four respectively. All four `handover_block` scenes
succeeded only with native precision. All 80 trace/action records, paired initial
inputs and frozen source checks passed. These small paused-simulation screens
establish neither population loss, non-inferiority nor real-time reliability.
[Verified pairs and protocol](VLA_JOINT_SCREEN.md).

## Current VLA V2 closed-loop screen

All 40 paired RoboTwin scenes completed on Thor. Clean measured native **17/20**
and FP8 **18/20** (+5 percentage points); randomized measured **16/20** and
**18/20** (+10 points). Clean has one native-only and two FP8-only successes;
randomized has zero and two respectively. All initial input pairs, wire/controller
traces and final source checks passed. These small paused-simulation screens
establish neither superiority, non-inferiority nor real-time reliability.
[Complete pairs and protocol](VLA_JOINT_SCREEN.md).

## Current public Runtime: pi05 closed-loop screen

The current v044 50-action Runtime completed all 20 fixed LIBERO-10 scene pairs:
native **16/20**, FP8 **17/20** successes, an observed +5-point difference.
There are **3 FP8-only successes and 2 native-only successes**. All episode wire
traces/actions and initial-observation pairing passed verification. This is a small
closed-loop screen, not evidence of zero loss, non-inferiority or superiority.
The simulator pauses for each reply; these outcomes do not qualify real-time tails.
[Verified pairs](pi05-public-campaign-comparison.json) /
[protocol and reproduction](PI05_PUBLIC_SIM_SMOKE.md).

## Current GR00T LIBERO fine-tune screen

The official `libero_10` fine-tune completed all 20 paired scenes on Thor:
native **20/20**, FP8 **17/20**, an observed **−15-point** difference. All three
discordant pairs succeeded only with native precision. These are matched frozen
initial scenes with native processing and the same four-step schedule. The
sample does not establish population loss or non-inferiority, and the DROID
latencies above must not be substituted for this fine-tune.
[Verified paired report and protocol](GROOT_LIBERO_SCREEN.md).

## Historical controlled pi05 experiment

Current **pi05 base** has its own public pair: native parameters remain FP32,
with capture admitted by the exact startup check (318 replays). Fixed-state
generation p50 is **1218.24 → 76.11 ms** (16.01×); varying-state p50 is
**1215.98 → 74.03 ms** (16.42×). Both arms preserve the checkpoint's 50-action
queue and 32 output dimensions; all 1650 retained actions per arm are finite.
This is one process per arm with sixteen generated chunks per phase. FP8 initial
generation costs **5.54 s**, and varying-state maximum is **781.50 ms**, so hot
medians do not establish real-time reliability. Setup is separate (179.21 s
native, 4.61 s FP8). The native dtype differs from the BF16 LIBERO fine-tune, and
these whole-runtime ratios do not isolate FP8 arithmetic or certify task quality.
[Base-checkpoint measurements](pi05-base-paired-latency.json).

The frozen comparison used ten-action chunks, ten denoise steps and staged inputs,
with vision and prefix recomputed every call. Three fresh processes, 128 measured
calls per process; values are medians of process percentiles.

| Execution | p50 ms | p99 ms |
| --- | ---: | ---: |
| Native BF16 eager | 311.78 | 323.70 |
| Native BF16 capture | 221.73 | 222.71 |
| Engine FP16 | 79.97 | 80.65 |
| Engine FP8 | 44.89 | 45.05 |

FP8 is 6.95× faster than eager, 4.94× faster than native capture, and **1.78×**
faster than the FP16 engine. The last comparison is the closest available control
for the incremental gain from FP8. Engine FP16 already changes BF16 arithmetic;
its gain is not bit-exact acceleration. Native capture actions matched eager in
these samples; FP8 maximum normalized action difference was 0.153053.

These ten-action results do not describe the checkpoint-native 50-action public
API. The current API preserves native processors, state tokens, camera masks,
50-action buffering and reset. Cached FP8 generation measured 61.26 ms p50 with
fixed state and 53.87 ms with varying state, but a first-seen prompt length reached
661.60 ms. Initial generation took 3.61 seconds. Cache-warm medians do not guarantee
a real-time deadline.

The current native capture API check completed with the same observation schedule
and 50-action chunks: fixed-state generation p50 **339.05 ms**, varying-state
**339.12 ms**. FP8 is respectively **5.54×** and **6.30×** faster at these medians.
Each phase contains only eight generated chunks in one process. Native fixed-state
p99 was 563.36 ms (maximum 579.78 ms); FP8 varying-state p99 was 659.84 ms.
Neither median ratio certifies deadline reliability.

Native capture recorded 158 denoise graph replays, passed its exact startup check,
and all 850 retained decoded actions equal the earlier native eager run. Native
parameters remain BF16/FP32; both precision arms have TF32 and cuDNN benchmark off.
Runtime setup was 166.19 seconds native and 3.84 seconds FP8; first generation
adds 0.81 and 3.61 seconds respectively. Buffered single-action replies measured
about 4.2–4.3 ms native and 1.4 ms FP8. Setup and buffer reads are separate from
chunk-generation timing. [Current pi05 receipts](pi05-public-comparison.json).

Full historical controls: [frozen comparison](../fp8_comparison_2026-09-09/README.md).
It describes its build at measurement time; later API fixes do not rewrite it.

## Current public Runtime: Cosmos DROID

The latest same-boot six-arm study supersedes the older timings below for current
performance comparison. Edge measured native **3454.31 ms**, current FP8
**3766.91 ms**, and a scale-reduction candidate **3561.41 ms**. Nano measured
**10321.11 / 10967.70 / 10392.84 ms** respectively. The candidate reduces current
FP8 latency by **5.46% / 5.24%**, with byte-identical retained FP8 actions for both
models, but remains **3.10% / 0.70% slower than native**. The allocation optimization
has now been merged into the local production module; its direct production-module
Thor regression passes with a matching source hash. The measurements above retain the candidate source
hash and do not establish FP8/native equality or task quality. All arms use 32 actions, four
UniPC steps, CFG 3 and FPS 15, with two warmups and 12 measured calls per process.
[Latest matched receipts and action hashes](cosmos-scale-study-comparison.json).

The following results retain the earlier implementation history:

The completed matched campaign uses native DROID processing, 32-action chunks,
four UniPC steps, CFG 3 and conditioning FPS 15. Both arms disable TF32 and
cuDNN benchmark. Native runs eager; FP8 replaces selected attention Q/K/V
projections. Each arm has one process, two warmups and 12 measured calls with
alternating recorded cameras/prompts and synthetic states.

| Model | Native p50 ms | FP8 p50 ms | FP8 latency change |
| --- | ---: | ---: | ---: |
| Edge | 3440.27 | 4057.02 | **+17.9% (slower)** |
| Nano | 10194.79 | 11700.22 | **+14.8% (slower)** |

These implementations execute FP8 successfully but do not provide an acceleration
benefit. The campaign does not qualify the best possible native implementation,
sustained tails or task success. All four retained action arrays were locally
verified finite with shape 15×32×8. Setup and first generation are separate in the
receipts. [Measurements and action hashes](cosmos-public-latency-comparison.json).

The subsequent fused-packing Edge run reduced FP8 p50 from 4057.02 to
3766.81 ms (7.2%), retaining exactly the same 15 action chunks as the prior FP8
run. It remains 9.5% slower than native. Nano FP8 p50 falls from 11700.22 to
10910.27 ms (6.8% lower), still
7.0% slower than native. Its 15 retained chunks also match
the prior FP8 run exactly. These runs precede the NaN-propagation correction.
[Retained-action check and measurements](cosmos-fused-packing-comparison.json).

## Current public Runtime: VLA V2

Matched pinned RoboTwin checkpoint, native processors, live vision, ten denoise
steps and 50×14 decoded actions. Each arm has one process, four warmups and 32
measured calls, including prompt resets at calls 13 and 25.

| Execution | p50 ms | p99 ms |
| --- | ---: | ---: |
| Native eager | 764.29 | 790.18 |
| Native precision with NUMERIC capture/preprocessing | 423.50 | 432.21 |
| FP8 engine with native BF16 vision | 238.98 | 1029.45 |

FP8 is **3.20×** faster than eager and **1.77×** faster than captured native at
the median. Its first calls after the measured prompt resets took about 1.02–1.03
seconds, so the median does not establish deadline reliability. These short-run
p99 values are descriptive, not sustained-tail qualification.

The native/capture retained arrays (37×50×14) are exactly equal. Capture remains
NUMERIC because these samples do not replace its legacy admission guard or prove
all-input equivalence. Actual counters confirm 358 denoise graph replays and 34
vision/prefill replays. FP8 versus either native arm has decoded-action MAE
0.012277 and maximum absolute difference 0.267903; these mixed-unit values are
not success-rate loss. Native arms retain matmul TF32; FP8 disables it. All arms
disable cuDNN TF32 and benchmark. This is a whole-implementation comparison,
not an isolated FP8 arithmetic experiment.
[Verified measurements, actions and source hashes](vla2-public-comparison.json).

## VA LIBERO current closed-loop screen

The full-schedule public Runtime pair completed all 20 fixed LIBERO-10 scenes:
native **20/20**, FP8 **18/20**, an observed **−10-point** difference. Both
discordant scenes succeeded only with native. All 40 traces/actions, paired
initial cameras and final source/weight checks passed. This is a small paused
screen, not non-inferiority or real-time qualification.
[Verified report](VA_LIBERO_SCREEN.md).

## Other models and quality evidence

RoboTwin VA's completed full-25V/50A synthetic-history saturation pair measures
**8496.87 → 8150.24 ms (1.04×)** at first exposure and **8491.30 → 5614.70 ms
(1.51×)** with graph shapes established. All 48 corresponding reset outputs
match byte for byte within each precision, and all 368 source/config/input
hashes remained unchanged. This is repeated recorded history, not task-quality
or sustained-tail evidence. [Verified stress results](VA_SATURATION.md).

LIBERO-long VA now also has a completed 48-cycle synthetic-history stress pair.
In the conservative saturation interval, first exposure measures native
**4688.51 ms** versus FP8 **4707.49 ms**, effectively no speed gain. Repeating the
episode with graph shapes already established measures **4694.00 → 2596.81 ms**,
**1.81×**. All 48 corresponding reset actions match byte for byte within each
precision. These repeated recorded windows are not a physically continuous robot
episode, and neither phase qualifies task quality or sustained tails.
[Full paired receipts and protocol](VA_SATURATION.md).

LIBERO-long VA now has a matched full-schedule public API latency pair: native
**4160.92 ms**, FP8 **2078.53 ms** p50, **2.00×** speedup. Both arms use the same
recorded frames/executed actions, native 20V/50A schedule and original history
windows. Each has one warmup and three measured three-cycle episodes; all twelve
7×4×4 chunks are finite and repeated episode outputs are byte-identical within
each precision. Native/FP8 action MAE is 0.006699, max 0.082921 (not success-rate
loss). This measures early history, not saturated KV rings or sustained tail
latency. [Paired measurements](va-libero-paired-latency.json).

RoboTwin VA's corresponding full-25V/50A pair measures native **5564.59 ms** and
FP8 **2815.17 ms**, **1.98×**. All twelve 16×2×16 chunks per arm are finite and
repeat byte-identically within each precision. Native/FP8 MAE is 0.002581 and
max difference 0.023438; these are native output-tensor deltas, not task-success
loss. The same short, early-history limitations apply.
[RoboTwin paired measurements](va-robotwin-paired-latency.json).

The latest GR00T weight-cast cache reduces FP8 p50 from **133.93 to 127.40 ms**
(4.9%), with all retained actions byte-identical to the previous FP8 build. It
retains about **447 MiB** of FP32 weight conversions; current observation features
and KV are still recomputed. Native in the new pair measures **110.30 ms**, so
FP8 remains 15.5% slower. [Weight-cache comparison](groot-weight-cache-comparison.json).

The preceding GR00T shared-native builder completed its matched public test:
native **108.80 ms**, FP8 **133.93 ms** p50. FP8 is 13.5% faster than the prior
FP8 build below, with byte-identical FP8 actions, but remains **23.1% slower than
native**. Both now retain the fast decoder and backbone metadata optimizations.
[Corrected comparison](groot-shared-native-comparison.json). A separate explicit
capture qualification passes its exact startup gate and retains byte-identical
actions, yet measures **114.38 → 122.68 ms**, 7.25% higher latency. Therefore the
production Thor planner keeps eager execution. [Capture results](groot-capture-qualification.json).

The following paragraph records the earlier build:

GR00T's completed DROID public comparison measured native 109.14 ms p50 and FP8
154.86 ms (+41.9% latency), with live visual processing in both arms. Raising
the native tier ceiling did not activate capture: that arm measured 109.85 ms
and retained byte-identical native actions. FP8 action MAE was 0.003421 and
maximum difference 0.037980, not task-success loss. The measured FP8 builder
omitted the native fast decoder and backbone metadata cache; the corrected
shared-native builder and strict capture qualification were measured subsequently
as reported above. [Preserved measured build](groot-public-comparison.json).

| Family | Available evidence | What it does not establish |
| --- | --- | --- |
| VLA V2 | Current matched public native/capture/FP8 speed and action comparison; complete 40-pair RoboTwin screen reported above | Sustained tails or non-inferiority; the small screen does not establish population loss, and historical native capture itself had nonzero action deltas |
| GR00T | Corrected DROID shared-native builder and explicit capture measured; separate LIBERO fine-tune screen completed at 20/20 native versus 17/20 FP8 | DROID FP8 remains slower; its timings do not describe the LIBERO fine-tune. Six embodiments have input/reset checks, while XDof statistics, DROID task quality and non-inferiority remain unresolved |
| LingBot VA RoboTwin | Full 25V/50A early-history pair and complete 48-cycle saturation stress; first-exposure 1.04× / shape-warm 1.51× at saturation | Sustained tails or current closed-loop certificate; stress uses repeated recorded history |
| LingBot VA LIBERO-long | Full 20V/50A controls and saturation stress; complete current Runtime LIBERO screen: native 20/20, FP8 18/20 | Sustained tails and non-inferiority; synthetic-history timing and paused task outcomes do not certify real-time operation |
| Cosmos Edge/Nano | Native/FP8 public Runtime passes independent camera/state/prompt, reset and repeat checks using original DROID service, 32-action chunks, CFG3 and FPS15 | FP8 is slower in the matched short API test above; no current task-quality certificate; historical 16-action RoboTwin-service timings use a different contract |
| DreamZero | Both native/FP8 complete two three-cycle episodes with identical repeats within each precision; all 120 FP8 Q/K/V projections executed and 1317 native loaded DiT tensors equal the checkpoint | Cross-process reproducibility, controlled speedup or task quality |

Current execution receipts and scope are recorded in the [completion report](README.md).
API checks establish execution and input/reset behavior; they do not establish
speed superiority or simulator success rates.

DreamZero's retained native/FP8 controls have decoded-action MAE 0.019402 and
maximum absolute difference 0.405273. Repeat-episode calls were respectively
20.53/23.42/24.55 seconds and 21.06/23.70/25.01 seconds; this diagnostic does not
show an FP8 acceleration benefit. There is one process per arm, and native ran
after FP8 populated compiler caches, so first-call timings are not comparable.
[Actions, diagnostic timings and limitations](dreamzero-native-fp8-controls.json).

Historical paired simulation observed pi05 85.6% → 84.4% (500 episodes, −1.2
percentage points) and V2 89.36% → 87.64% (1100 episodes, −1.73 points). Both passed
their declared five-point non-inferiority tests. These are older recipes and
cannot certify the corrected current Runtime, or imply zero loss.

Default deployment should retain native precision when measured end-to-end tails
meet the application deadline. Opt-in FP8 needs a paired evaluation of the actual
checkpoint, calibration, processors and schedule. Total device memory, energy per
action and long-duration thermal behavior remain unmeasured here. The historical
pi05 allocator peaks (native 8.804 GiB, FP8 3.907 GiB) exclude external allocations
and must not be advertised as total memory savings.
