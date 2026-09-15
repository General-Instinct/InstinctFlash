# Thor native and opt-in FP8 completion

Status: active implementation. This is not a completed all-model support claim.

VA LIBERO Runtime now completes its verified 20-pair screen: native **20/20**,
FP8 **18/20** (observed −10 percentage points), with two native-only successes.
All traces, paired initial inputs and final source/weight checks passed.
[Final report](VA_LIBERO_SCREEN.md).

VLA 4B/V2 native and FP8 endpoints passed real-input Thor startup admission.
VLA 4B completed all 40 paired RoboTwin scenes: clean native **15/20**, FP8
**14/20**; randomized native **15/20**, FP8 **16/20**. All paired traces and
initial observations passed verification. All four `handover_block` scenes
succeeded only with native precision. These small screens do not establish
population loss or non-inferiority. V2 also completed all 40 pairs: clean
native **17/20**, FP8 **18/20**; randomized native **16/20**, FP8 **18/20**.
All paired traces, initial inputs and final source checks passed; these results
do not establish superiority.
[Paired results, protocol and startup evidence](VLA_JOINT_SCREEN.md).

The current pi05 v044 public Runtime completed a paired LIBERO-10 screen on Thor:
native **16/20 successes (80%)**, FP8 **17/20 (85%)**, with 3 FP8-only and 2
native-only successes. All 40 episode traces and paired initial observations
passed verification. Native preprocessing, ten denoise steps and the 50-action
queue remain intact. This small screen does **not** establish zero loss,
non-inferiority, superiority or real-time reliability.
[Paired results](pi05-public-campaign-comparison.json) /
[protocol and replay setup](PI05_PUBLIC_SIM_SMOKE.md).

The official GR00T LIBERO-10 fine-tune also completed a Thor paired screen:
**native 20/20, FP8 17/20**, with three native-only successes (observed −15
percentage points). All 40 results passed verification; the three unsuccessful
FP8 episodes remain in the report. This does not certify non-inferiority or
estimate a universal quantization loss. [GR00T paired results](GROOT_LIBERO_SCREEN.md).

The frozen original DreamZero native source has now been rerun: its six action
chunks equal the latest native run byte for byte, while differing from the
earliest native run by the same amount. Thus the new feedback validator is not
required to produce the observed native drift. The underlying run-state cause
is still unresolved. [Original-source control](dreamzero-frozen-native-recheck.json).
A same-process new/old/new FP8 scale-reduction control now passes: nine retained
chunks are byte-identical by cycle with one loaded model and identical packed
weights, and all 120 projections execute 162 times. This isolates the scale
change for these inputs; it does not establish universal model equivalence or
explain cross-process drift. [Scale isolation control](dreamzero-scale-control.json).

Latest DreamZero native source passes public execution, feedback refusal before
prediction, repeat-episode byte equality and cleanup. However, its retained
actions differ from the earlier native run (MAE 0.008603, max 0.056961 in mixed
action units). The cause is unresolved; an original-source recheck is queued to
separate code changes from other run-state effects. Do not claim cross-run native
bit-exactness. [Current native discrepancy](dreamzero-current-native.json).
The current FP8 arm also completes all six finite chunks, exact episode repeats,
120 projections × 108 calls, feedback refusal and cleanup, but differs from its
earlier FP8 run (MAE 0.026064, max 0.422852). The current native/FP8 pair has MAE
0.010684 and max 0.071856. These mixed-unit action deltas do not measure task
quality; source or run-state causes remain under investigation.
[Current FP8 receipt](dreamzero-current-fp8.json).

The latest matched Cosmos scale study completes both sizes. Reducing activation
maxima in BF16 and widening only the scalar to FP32 lowers FP8 median latency by
5.46% (Edge) and 5.24% (Nano), retaining byte-identical FP8 actions for all 15
chunks per size. It still does not beat native execution. Exhaustive BF16 value
group checks and full linear/reference/CUDA Graph checks passed for the candidate.
The same reduction is now in the local production module; its direct Thor
regression passes with an identical source hash, including reference outputs,
all BF16 value groups and changed/restored CUDA Graph inputs. Nineteen related
integration/unit checks also pass. [Production operator receipt](fp8-scale-production.json).
[Matched study](cosmos-scale-study-comparison.json),
[candidate operator checks](fp8-scale-candidate.json).

GR00T's shared-native FP8 builder now measures 133.93 ms p50 versus 108.80 ms
native, reducing the prior FP8 latency by 13.5% but remaining 23.1% slower than
native. All FP8 actions equal the prior FP8 build byte for byte; live backbone,
fast decoder and metadata cache are verified in both arms.
[Corrected builder comparison](groot-shared-native-comparison.json).
Separately, explicit native capture passes its strict six-input startup gate and
all retained actions equal eager, but measures 122.68 versus 114.38 ms in its
paired control (+7.25%). Production Thor capture remains disabled because this
test does not demonstrate a speed benefit. [Capture qualification](groot-capture-qualification.json).

V2's repaired SDPA capture now completes the matched public native/capture arms:
p50 764.29 → 423.50 ms (1.80×), with all 37 retained 50×14 action chunks exactly
equal. Actual graph counters record 358 denoise and 34 vision/prefill replays.
The capture tier remains NUMERIC; this sample equality does not replace the
legacy admission guard with a universal bit-exact guarantee. FP8 pairing also
completed: p50 238.98 ms, 1.77× faster than capture. Prompt-reset calls exceeded
one second, and decoded-action MAE is 0.012277 versus native; neither result
certifies task quality or real-time deadlines.
[Verified three-arm comparison](vla2-public-comparison.json).
The earlier capture failed because SDPA copied sequence lengths to CPU inside
CUDA Graph capture. The fix hoists fixed-grid metadata to CPU and retains native
attention and FA2's metadata device. Four CPU regression tests pass alongside
the actual device run. [Preserved failure receipt](vla2-sdpa-capture-interrupted.json).

DreamZero's direct-BF16 loading candidate now initializes the full native policy
on Thor after resolving the native `av==15.0.0` dependency. All 1317 loaded DiT
tensors equal the original checkpoint before FP8 mutation.
[Actual loaded-value receipt](dreamzero-thor-loaded-values.json).
The direct native-adapter-plus-FP8 check now completes two three-cycle episodes:
all six 24×8 action chunks are finite, both episodes match exactly, history
positions repeat 3/5/7, and all 120 FP8 projections execute 108 times each.
[Locally verified history receipt](dreamzero-thor-fp8-history.json).
The first call took approximately 310 seconds including native first-execution
work; repeat-episode calls are faster, but these are not matched speedup results.
The native control also completes both episodes with exact repeatability and
1317 loaded DiT tensors matching the checkpoint. FP8 versus native decoded-action
MAE is 0.019402 and maximum difference 0.405273 (mixed units, not task-quality
loss). Repeat-episode native calls took 20.53/23.42/24.55 seconds, FP8
21.06/23.70/25.01 seconds. This short diagnostic does not demonstrate an FP8 speed
benefit; native ran after FP8 populated compiler caches, so first-call timings
are not comparable. [Retained controls](dreamzero-native-fp8-controls.json).
The local adapter now owns the low-memory checkpoint view for full-architecture
builds; construction failures and close release it. Public FP8 dispatch is now
implemented with actual-build precision, step-count and CFG checks; 70 local
dispatch/configuration tests pass. The adapter entry point is installed on Thor,
and native/FP8 public API runs are queued. The actual updated FP8 builder now
passes both episodes; all six chunks match the earlier FP8 implementation byte
for byte, all 120 projections execute, and close removes its owned checkpoint
view. [Builder and cleanup receipt](dreamzero-thor-owned-builder.json).
The native public-interface run now passes all six 24×8 chunks, repeat-episode
byte equality, history positions 3/5/7 twice, zero FP8 projections and owned-view
cleanup. [Public native receipt](dreamzero-public-native.json). The FP8 public arm
also passes all six chunks, repeat byte equality and cleanup; all 120 projections
execute 108 times each. Its actions equal the previous FP8 control byte for byte.
[Public FP8 receipt](dreamzero-public-fp8.json). These runs use the frozen source
before the latest optional feedback validator and scalar-reduction optimization;
they establish execution for that build, not task quality or a speedup. Separate [strict byte checks](retained-action-byte-checks.json)
also confirm the retained V2 native/capture and DreamZero within-precision repeats.

LIBERO-long VA now passes actual-checkpoint public Runtime checks in both
native and FP8 modes: full 20V/50A, checkpoint action shift 0.05, native T5/VAE
and processing, two three-cycle episodes with history positions 0/4/8 and
identical repeated-episode actions within each precision. All 16 checkpoint
files passed SHA256 verification. [Paired numerical receipt](va-libero-public-numerical.json).
This recorded-frame/executed-history check is not a closed-loop success-rate or
qualified speed certificate. The public native/FP8 independent-control campaign
now passes on Thor: each camera, prompt and executed history independently change
actions; reset restores the baseline action bytes. Future execution feedback does
not alter the current prediction. Both arms retain 20V/50A and the original
12-frame first history window; nine finite 7×4×4 action arrays per arm were
independently verified. [Verified input controls](va-libero-independent-inputs.json).
The [probe](probe_va_libero_public_inputs.py) is an execution test, not a simulator
success-rate or sustained-latency evaluation.

The separate full-20V/50A paired latency test now measures native 4160.92 ms and
FP8 2078.53 ms p50 (2.00×) on identical recorded frames/executed history. All
twelve chunks per arm are finite and repeat byte-exactly across episodes within
each precision. One episode warms the paths and three episodes are measured;
these are early-history timings, not saturated-ring or long-tail qualification.
[LIBERO-long paired latency](va-libero-paired-latency.json).
The RoboTwin full-25V/50A pair also completes: native 5564.59 ms versus FP8
2815.17 ms p50 (1.98×), with all twelve finite 16×2×16 chunks repeating byte for
byte within each precision. It uses the same early-history protocol and does not
qualify saturated KV rings or task success.
[RoboTwin paired latency](va-robotwin-paired-latency.json).

Cosmos DROID update: actual Edge and Nano native/FP8 service checks pass using
NVIDIA's RoboLab processing, 32-action chunks, FPS 15 and CFG 3. Edge uses JSON
prompts; Nano uses plain prompts. The [Edge public FP8 Runtime check](cosmos-public-edge-fp8.json)
also passes repeat/reset/input-change checks. Individual camera/state/prompt
public checks now pass for both sizes and precisions: each arm retains six
32x8 decoded chunks, verifies independent input sensitivity, and restores
identical actions after reset. Receipts:
[Edge native](cosmos-public-inputs-edge-native.json),
[Edge FP8](cosmos-public-inputs-edge-fp8.json),
[Nano native](cosmos-public-inputs-nano-native.json),
[Nano FP8](cosmos-public-inputs-nano-fp8.json). These checks do not
establish speed superiority or task quality. Historical RoboTwin-service timings
used a different input/gripper/prompt and 16-action contract and cannot certify
this corrected DROID implementation.

Thor FP8 projection prerequisite: the [operator receipt](torch-fp8-linear.json)
and [reproducible probe](probe_torch_fp8_linear.py) validate BF16 input/output with
E4M3 weights/activations, with and without bias, empty token partitions,
noncontiguous inputs, zero inputs/weights and CUDA Graph changed/restored inputs.
This is an operator check for ongoing Cosmos integration, not model support,
performance or task-quality evidence. Native execution does not select it.

See [the precision comparison](COMPARISON.md) for current public-API timings,
historical arithmetic controls, quality evidence and their distinct limitations.

The goal is one user-facing Runtime/serve interface for every supported model on
Thor. Default execution preserves the checkpoint's native precision, geometry,
step schedule and observation/action semantics. `--fp8` explicitly selects actual
FP8 execution, with the same external contract. Numerical and closed-loop quality
evidence remain separate from API support and performance.

## Completion requirements

- `--fp8` and `precision="fp8"` select the same declared arithmetic policy;
  absence of the flag keeps native precision. Configuration and conflicting
  command-line choices must be unambiguous.
- Each model has real native and FP8 implementations on Thor. Merely expanding a
  registry, returning synthetic actions, leaving a stub, silently serving native
  arithmetic under FP8, or refusing the model does not meet the goal.
- Both implementations preserve camera order/masks, state and per-call prompt,
  checkpoint normalization, action dimensions/horizon, buffering, reset and
  model-specific history/commit semantics. No implicit distillation or shorter
  denoise schedule.
- FP8 computation and calibration are identifiable; the native path does not
  implicitly quantize. Quality and capture admission are accurately reported.
- Verify each model through the actual public API on Thor with changing real
  observations, episode reset and repeated calls. Compare matched timing scopes,
  numerical outputs and appropriate paired simulator/task evidence. A unit test
  or one checkpoint cannot certify the full model list.
- Document reproducible setup and precision choice, with an evidence matrix that
  reflects actual current execution, not inherited historical certificates.

## Full model scope

VA saturation follow-up completed separately from the published early-history
latencies. The probe drives 48 cycles per episode, two episodes per precision and
family, preserving native schedules and explicitly repeating the last recorded
observation/action window. It records live cache occupancy, frame positions and
every action chunk; the second episode checks reset repeatability. This is
synthetic-history stress, not a physically continuous episode or task-quality
test. Both pairs are complete and verified. RoboTwin saturated first exposure measures 1.04× and graph-shape-warm repeat measures 1.51×; see [the full report](VA_SATURATION.md). [Frozen protocol](va-saturated-stress-protocol.json)
and [probe](probe_va_saturated_stress.py).
The [comparison verifier](compare_va_saturated_stress.py) requires complete
96-call arms, matching schedules/boot/source receipts, retained action hashes,
finite native-shaped outputs, exact episode resets and matching normalized cache occupancy. Native terminal
elision defers provisional action-slot reservation; FP8 reserves those slots
immediately, so the comparison subtracts those FP8-only post-call slots. This
checks occupancy counts, not token identities.
It identifies the conservative eviction regime using token growth and the
post-transient occupancy plateau; end-of-call occupancy need not equal capacity.
Seven verifier tests cover both checkpoint capacities, the expected plateau,
first-exposure versus shape-warm latency, and semantic corruption with
otherwise matching artifact hashes. The LIBERO pair is verified; RoboTwin remains pending. A [source manifest](va-stress-source-manifest.json)
pins 368 source, binary, configuration and input files from the running campaign;
its probe/input hashes match the launch protocol. Paired latency reports
the first-exposure episode separately from the shape-warm repeat; the FP8
frontend keys CUDA Graphs by stream, spans and history positions, so warming only
early cycles does not remove the cost of first encountering later positions.
LIBERO native has now completed all 96 calls: every corresponding action in the
two 48-cycle episodes is byte-identical, and cache/history/source checks pass.
The conservative saturated interval (zero-based cycles 15–47) measures 4694.00 ms
median and 4911.98 ms maximum in the measured episode. This is a synthetic-history
stress result, not a sustained-tail or task-quality certificate. The completed LIBERO FP8 pair measures 4707.49 ms on first exposure (no speed
gain versus native), then 2596.81 ms after shape warmup (1.81× versus native).
Both arms pass all 48 reset-repeat checks. RoboTwin remains in progress.
[Full saturation report](VA_SATURATION.md). [Verified native summary](va-saturated-libero-native-verified.json)
and [full arm receipt](va-saturated-stress-libero-native.json).

GR00T embodiment follow-up: the FP8 frontend previously selected slots from an
incomplete built-in table. It now reads `embodiment_id.json` from the checkpoint,
including custom slot assignments, and validates indices against loaded weights.
All 52 entries in the pinned Thor checkpoint's manifest resolve correctly in a
CPU check; constructor/validation and related GR00T regressions pass (20 tests).
The revised frontend passed the DROID native/FP8 public Thor regression: seven
calls per arm, independent camera/state/prompt changes and byte-identical reset
restoration within each precision. FP8 replay and checkpoint slot 24 verified.
[DROID pair](groot-embodiment-droid-pair.json) /
[manifest evidence](groot-checkpoint-embodiments.json).
Additional embodiments remain incomplete: the subsequent XDof subtask native run
failed in upstream action decoding. Its checkpoint has empty absolute wrist EEF
statistics, and upstream retains zero action widths when replacing them with
nonempty relative statistics. This produces a `(1,40,0)` versus `(40,9)` mismatch.
The campaign stopped on this failure; no later embodiments are claimed tested.
The follow-up campaign continues the previously unattempted G1/R1 variants
separately. G1 now passes both native and FP8 public Thor checks: six calls per
arm, one camera, native 40×53 actions, changed camera/state/prompt responses and
byte-identical repeat/reset restoration within each precision. Retained action
hashes and arrays were independently checked locally.
[G1 pair](groot-embodiment-real_g1_relative_eef_relative_joints-pair.json).
R1 base now also passes both precision arms: eight calls per arm, three
independently changed cameras and native 40×62 actions, with all retained arrays
verified finite and repeat/reset restoration byte-identical within each precision.
[R1 pair](groot-embodiment-real_r1_pro_sharpa_relative_eef-pair.json).
The R1 probe uses declared `XYZ_ROT6D` action reference fields to construct valid
synthetic poses; its earlier name-based initializer supplied invalid zero
rotations for R1 and that failed attempt remains preserved. All four R1 variants have now completed both precision arms; all action files
were verified locally. Together with DROID and G1, six embodiments pass, totaling
82 retained action chunks. See [the embodiment report](GROOT_EMBODIMENTS.md) for
individual receipts, reproduction and the two unresolved XDof configurations.
The [checkpoint statistics audit](groot-statistics-audit.json) confirms that both
XDof variants have empty wrist state and absolute-action statistics while the
processor requires state (`exclude_state=false`); the six other statistics-backed
variants do not have that particular defect. This file audit does not certify
inference for untested variants.
The optional DROID fast decoder now skips checkpoints without a DROID modality
configuration, keeping their native decoder. This resolves a custom-fine-tune
loading dependency; 15 related selection/mapping/builder tests pass. The live
embodiment campaign retains its frozen source overlay, so its receipts do not
claim to measure this subsequent guard change.
Also, the pinned upstream public `Gr00tPolicy._get_action` ignores its `options`
argument and calls the model without RTC inputs. Existing internal action-head
RTC tests therefore do not establish public Runtime RTC support.

The initial inventory is every checkpoint in `KNOWN_DECLARATIONS`. Other declared
variants and compatible user fine-tunes must preserve their own configuration;
support is not restricted by hardcoded public checkpoint IDs.

| Checkpoint | Current evidence and remaining work |
| --- | --- |
| lerobot/pi05_base | Native FP32 capture/FP8 public 50-action pair measured (16.01× fixed-state median); cold/profile tails and task evaluation remain |
| lerobot/pi05_libero_finetuned_v044 | Native/FP8 public API, 50-action latency and 20-pair LIBERO screen complete (16/20 vs 17/20); statistical quality certification and sustained tails remain |
| robbyant/lingbot-vla-4b-posttrain-robotwin | Corrected BF16-vision FP8 API and latency pair checked; 40-pair RoboTwin screen complete: clean 15/20 → 14/20, randomized 15/20 → 16/20. All four handover_block scenes succeeded only with native. Non-inferiority and sustained tails remain |
| robbyant/lingbot-vla-v2-6b-robotwin | Native/capture/FP8 public checks and latency measured; 40-pair RoboTwin screen complete: clean 17/20 → 18/20, randomized 16/20 → 18/20. Non-inferiority and sustained prompt-reset tails remain |
| nvidia/GR00T-N1.7-3B | Public DROID native/FP8 and weight-cast cache measured; FP8 still slower. Six embodiments pass native/FP8 input/reset checks; XDof statistics, public RTC and current task quality remain |
| robbyant/lingbot-va-posttrain-robotwin | Public native/FP8 full 25V/50A with identical recorded history measured, 1.98× early-cycle gain; saturated first exposure 1.04× / shape-warm 1.51×; task quality remains |
| robbyant/lingbot-va-posttrain-libero-long | Native/FP8 independent camera/prompt/history/reset controls and full 20V/50A pair pass, 2.00× early-cycle gain; saturated first exposure 1.00× / shape-warm 1.81×; task quality remains |
| nvidia/Cosmos3-Edge-Policy-DROID | Native/FP8 original DROID public controls and matched latency measured; FP8 slower, native optimization and task quality remain |
| nvidia/Cosmos3-Nano-Policy-DROID | Native/FP8 original DROID public controls and matched latency measured; FP8 slower, native optimization and task quality remain |
| GEAR-Dreams/DreamZero-DROID | Actual native/FP8 public history, feedback refusal, reset and cleanup verified. Same-process scale change preserves action bytes; cross-process reproducibility cause, matched performance and task quality remain |

## First implementation batch

- Added `--fp8` / `--fp8=false` to the shared typed runtime CLI parser. CLI
  overrides file precision; contradictory explicit CLI choices fail. Existing
  precision permission gates remain in force until real integrations land.
- Added explicit `action_chunk` and `action_output="normalized"` to the pi05
  Thor frontend. Normalized mode returns all 32 model dimensions and does not
  load unrelated LIBERO normalization statistics. Legacy standalone defaults
  remain available. Runtime now uses checkpoint geometry and its native processors.
- Real Thor probe: chunk 50 produced finite 50×32 outputs with active CUDA
  graphs, stable fixed-input repeats and changed actions for changed observations.
  This verifies execution geometry, not task quality or full API interchangeability.
- Existing CLI, runtime precision, geometry and operating-point test suites pass.

## Public pi05 interface integration

`instinctflash/runtime/pi05_engine.py` reuses checkpoint-native LeRobot processors
around normalized 32-dimensional engine output. It preserves state-containing
token IDs, active camera order/masks, native action decoding, `n_action_steps`
buffering and episode reset. Both published pi05 checkpoints compute 50-action
chunks; replies retain their respective 32- and 7-dimensional action spaces.

The Thor checks already completed include:

- `pi05-base-runtime.json` (rechecked after caching in `pi05-base-runtime-cached.json`):
  public FP8 Runtime, 32-dimensional replies, full
  50-action buffering, state/prompt changes, reset and two/three active views.
- `pi05-runtime.json`: corresponding public FP8 checks for LIBERO v044 with
  7-dimensional replies. `pi05-libero-runtime-replay.json` repeats these checks
  with retained action arrays and source/action hashes.
- `pi05-base-native-api.json` and `pi05-libero-native-api.json`: public default
  Runtime retains native BF16/FP32 parameters without FP8; 50-action buffering,
  refill and reset pass for both checkpoints.
- `pi05-prompt-padded-attention.json`: odd/even active token lengths run on Thor;
  same-length token updates reuse the graph, change actions, and restore the
  original actions when restored. `pi05-prompt-padding-invariance.json` also
  perturbs alignment-only rows and verifies that actions remain identical.
- Six local test scripts pass: CLI precision selection, Runtime precision,
  engine geometry, operating-point gates, pi05 processing/queue behavior and
  Thor attention dispatch. Receipts: `integration-tests.json` in the raw root.

Logical attention length is separate from aligned GEMM storage. Odd key counts
use the existing padded-logits kernel; padded language rows are excluded from
attention. Normalized mode does not reuse the legacy calibration cache, whose
key lacks the new geometry/masking identity. It currently serves one observation
with no CFG. A bounded prompt-length cache now retains graph/calibration allocations and
restores logical RoPE offsets. Time-conditioning tables are shared. New lengths
still require setup, so this is not a completed real-time deployment profile.

`probe_pi05_api_latency.py` measures actual public API calls, separating chunk
generation from buffered replies and exposing varying-state setup overhead.
First FP8 public-API run (`pi05-libero-fp8-api-latency.json`, one process,
eight generation calls per measured phase): fixed-state generation p50 **60.90 ms**,
varying-state generation p50 **368.83 ms**; buffered replies take about 1–1.4 ms.
Initial generation includes lazy engine setup and takes **3.71 s**, separately from
Runtime setup. The changing-length path rebuilds time tables, calibration and
graphs. This is a deployment performance gap to fix, not an FP8 speedup claim.
The matching default-native run completed: fixed-state generation p50 **408.85 ms**,
varying-state p50 **405.38 ms** (also one process/eight generations per phase).

After prompt-profile caching (`pi05-libero-fp8-api-latency-cached.json`), FP8
fixed-state p50 is **61.26 ms**, varying-state p50 **53.87 ms**. New-length setup
still reaches **661.60 ms**; the median must not hide this deadline risk. The
revised recipe also calibrates each new length on its first real observation,
then refreshes the visual prefix overwritten by calibration/capture warmup.
Its first-action outputs therefore cannot inherit the prior build's quality claim.
`pi05-prompt-cache-prefix-refresh-b1.json` verifies graph reuse, byte-equal output
restoration across 49/48/37-token profiles, live same-length token updates and
successful cache eviction. The two failed precursor checks are retained.

It uses recorded images with synthetic states, not a simulator. Contract checks
and historical success rates do not certify this build's closed-loop quality.

The subsequent current native capture check completed at 339.05 ms fixed-state
and 339.12 ms varying-state generation p50. Its startup exactness check passed;
all 850 retained actions match the earlier native eager run. See the linked
comparison for tails, setup, graph receipts and the matched FP8 ratios. These are
short one-process checks, not sustained deployment qualification.

Raw work root: `/home/ubuntu/ifl_eval/thor_precision_completion_20260909`.
Thor work root: `/home/guanming/ifl_eval/thor_precision_completion_20260909`.
The earlier geometry probes and the frozen comparison campaign retain their own
source identities; later interface changes do not retrospectively certify them.

## VLA4 native-processor FP8 Runtime

`instinctflash/runtime/vla4_engine.py` preserves the upstream server's image
processing, FeatureTransform, state normalization, robot reset and 25x14 action
slicing. `vla4-runtime-native-vision.json` verifies the public FP8 Runtime with
fixed-input repeats and changed state, camera and prompt; the direct bridge
check is `vla4-engine-loop-native-vision-eager-replay.json`.

The earlier standalone FP16 vision path overflowed at the 18th block's down
projection on native-processed real camera images. NaN vision features then
produced finite but image-insensitive actions. This is recorded in
`vla4-vision-first-nonfinite.json` and the retained camera/vision diagnostics.
The new route retains the native **BF16 eager vision tower**, with **FP8 language
and action experts**. The engine rejects nonfinite visual features before the
language/expert path. This recipe now has short paired API timing in the linked
comparison, and still needs new task-quality evidence;
the historical standalone 98.94 ms cell cannot be attributed to it.

Six integration test scripts were run. Five passed unchanged; the operating-point
suite contained an obsolete assertion that VLA4 was not executable, updated to
reflect the real bridge. Its rerun passed (`vla4-final-operating-tests.log`). All
schedule/precision/exclusion gates remain, including refusal to serve an
unsupported NFE or silently use native math under `precision="fp8"`.

The Thor model environment now has the local `lingbot-vla-iwm` adapter installed
with `--no-deps --no-build-isolation`; the install log is retained in the raw root.
No model-stack dependency was upgraded. Native BF16 vision uses the qualified
Thor stack's eager attention, without requiring a separate flash_attn wheel.

Next: handle first-seen length setup/prewarming, measure both public precision
modes with matched inputs and geometry, and establish paired
closed-loop evidence. Integrate the remaining model families with their own native
processing and history contracts; the entire model scope above remains required.

## V2 live-vision staged integration

`instinctflash/runtime/vla2_engine.py` now supplies checkpoint-native BF16 Qwen3
vision, all three deepstack taps, and a staged chain with checkpoint-derived FP16
prefill and real FP8 action/MoE kernels. It does not use historical repack or
calibration artifacts. The public backend registry now includes V2 following
the native-processor bridge and public Runtime checks below.

`probe_vla2_native_vision.py` passed on two recorded Thor frames: all four feature
arrays are finite, change with the camera input and restore byte-for-byte on a
repeated frame. `probe_vla2_staged_engine.py` passed the complete vision/prefill/
ten-step expert chain with four graph replays and normalized 50×55 actions.
Observation changes affect actions; a separate camera-only change also affects
actions; restoring the original observation restores the original output exactly.

Calibration collects each layer's maximum activation scale across all ten denoise
steps, including routed MoE down scales. The dynamic kernels otherwise overwrite
these shared slots at each step, leaving only the last step's values. Collection
runs on the same CUDA stream and refuses incomplete or nonfinite scale sets.
The first replay restores actual caller noise after calibration/capture warmup.

This four-call execution check took 1200.86 ms including first-call calibration,
then 242.84, 217.47 and 217.09 ms. These are diagnostic timings, not matched
performance or closed-loop quality evidence. CPU tests cover early-step maxima,
incomplete/nonfinite calibration, active-token masks, noise preservation and
geometry rejection. The native sample_actions bridge and native server processing/
reset are now exercised by `probe_vla2_engine_loop.py`. Both direct
(`vla2-engine-loop.json`) and public Runtime (`vla2-runtime.json`) executions passed,
returning 50×14 actions with identical retained arrays across both routes. Five
calls cover fixed-input repeatability, state/camera changes, prompt reset and
graph replacement for a new prompt. V2's nested sharded checkpoint is resolved by
its native adapter instead of requiring a root-level `model.safetensors` file.
Precision, geometry, exclusion and baked-schedule test scripts pass. Other model
families remain in scope; current closed-loop quality is not established here.

The current default native API also passed (`vla2-native-api-toolchain.json`),
with BF16 parameters, 50×14 replies, repeated-input equality and changing state,
camera and reset prompt. Capture was **not active** in this default plan. Its
five diagnostic calls are not an accelerated-native latency benchmark.

The first native attempt failed because Triton's bundled assembler rejected
`sm_110a`, followed by an upstream error handler referencing an undefined logger.
That failed log remains retained. The native adapter now detects a system CUDA
assembler advertising `sm_110a` and fills absent Triton compiler overrides before
model construction, preserving explicit user overrides. The successful rerun used
`/usr/local/cuda/bin/ptxas`, matching the previously qualified native campaign's
compiler choice; actual paths are in `graph_stats.ptxas`. No alternative precision
or MoE implementation was selected as a fallback. Compiler-selection tests and
the full V2 adapter test script pass.

## GR00T cross-attention refresh

The frontend previously computed DiT cross-KV only on its first inference. New
observations therefore need an explicit refresh, and replacing KV tensor objects
alone would leave captured graphs reading the old addresses. The new
`update_backbone_features` method updates KV in place when text/image token counts
are unchanged, and invalidates graphs/attention descriptors when counts change.
`set_prompt` now uses this same handoff. Invalid features or nonfinite projected KV
are rejected before replacing the current observation.

`probe_groot_kv_refresh.py` passed on Thor using the actual 32-layer BF16 DiT and
CUDA graphs. Perturbing image backbone features changes actions; restoring them
restores actions byte-for-byte. Same-geometry updates preserve all 32 K/V addresses
and graph objects; adding a text token rebuilds attention and graphs successfully.
Both the first check and final rerun are retained; their action arrays match.
CPU tests also cover geometry invalidation and rejection without replacing the
last valid observation.

This uses captured setup auxiliary tensors and synthetic feature perturbations.
It is not a camera-to-action FP8 benchmark: live raw-camera processing and the
FP8 backbone pipeline still need to be connected, followed by public native/FP8
API and paired quality validation. The subsequent public integration is recorded
below; these component measurements themselves are not a public-API certificate.

## GR00T DiT weight precision correction

Inspection of the load path found that BF16 DiT GEMMs used weights which had first
been quantized to FP8 and then dequantized to BF16. That round-trip changes values
without providing FP8 GEMM execution. The DiT weight spec now loads checkpoint
weights and biases directly as BF16 (transposing matrices for the existing GEMM
layout). Other FP8 stage weights remain quantized; this change does not turn the
entire frontend into a native-precision implementation.

`groot-native-dit-refresh.json` verifies all **448** DiT tensors (32 layers × seven
matrices and seven biases) against the actual checkpoint, with exact BF16 value
equality. The actual graph refresh checks pass with these weights: changed image
features affect actions, restored inputs restore action bytes, and changed token
counts trigger a working graph rebuild. CPU tests ensure the DiT spec has no
quantizer and preserves BF16 values not representable in FP8.

Earlier GR00T refresh records used BF16 computation with FP8-rounded DiT weights;
their original artifacts remain unchanged. The corrected recipe has distinct
actions and source hashes and requires its own numerical/closed-loop evaluation.
This is an exact weight-loading check, not a full-policy BITEXACT certificate.

## GR00T executable FP8 VLSA stage

`serving/flash_rt/models/groot_n17/vlsa_runner.py` executes vlln and all four
VLSA layers through actual E4M3 weight/activation GEMMs, FP16 attention and CUDA
graphs. It calibrates the first supplied LLM-feature sample for each token count,
then recomputes every observation and refreshes the DiT cross-KV. A different token
count reallocates buffers, recalibrates and recaptures. It does not return the
calibration shadow output as the serving result.

`groot-vlsa.json` records the Thor check with actual FP8 VLSA followed by the
native-weight BF16 DiT. Changed image-token features affect both feature and action
outputs; restoring the original features restores both byte-for-byte. Changing
token count recaptures and runs successfully. VLSA versus the floating-point
calibration shadow has maximum absolute feature difference 1.27986 and RMSE
0.05717 on this sample; these are not task-quality margins or success-rate losses.

The probe derives staged LLM features from the existing shadow chain and recorded
auxiliary setup, then applies a synthetic feature perturbation. It is not a
live-camera or complete public Runtime evaluation. The later native camera/LLM
integration is described below; matched performance and paired quality remain.

## GR00T native-processor public Runtime

`instinctflash/runtime/groot_engine.py` preserves the native Gr00tPolicy and its
camera/history processing, live BF16 vision/LLM, normalization, embodiment and
action decoder. The action generator executes FP8 VLSA followed by native-weight
BF16 DiT. It preserves native VLSA attention over all tokens, then filters only
the DiT KV according to the backbone attention mask. Four denoise steps and
40×132 normalized actions are checked explicitly. Native DROID decoding returns
40×17 joined actions, split action fields and native info through the same loop.

`groot-engine-loop-embodiment.json` and `groot-runtime-registered.json` passed on
Thor, with identical retained actions across direct and public FP8 interfaces.
Five calls exercise repeatability, state/camera changes, reset and a changed
prompt. The native camera/LLM backbone executes on every call. The first failed
attempt used the native uppercase embodiment enum name as an engine slot name;
the bridge now takes the native policy's canonical enum value. Its failure log
remains retained. The first public attempt also exposed a missing adapter entry
point; the local package was installed with `--no-deps --no-build-isolation`,
without upgrading the model stack.

The final public rerun (`groot-runtime-close.json`) retains the same action arrays
and additionally verifies that `close()` restores the native action-head method,
breaking the temporary generator/head reference cycle.

`groot-native-runtime.json` independently verifies the default BF16 public API
with the same inputs and output contract. Its DiT capture is inactive; native
fast decoding and backbone preprocessing optimizations are active. No numerical
equivalence or latency ratio between the two precision modes is claimed here.

RTC/inpainting, when requested at the native action-head boundary, still runs
FP8 VLSA and delegates the special denoise schedule to the native loop. Unit tests
verify option/state/mask passthrough and embodiment rejection; real Thor RTC,
wider embodiment coverage, sustained performance and paired closed-loop quality
remain unqualified. GR00T is now in `ENGINE_BACKBONES` for this implemented route.


## VA checkpoint geometry and schedule prerequisites

The VA frontend now accepts `WanVaGeometry`, derived from the resolved native
configuration: RoboTwin uses F=2, latent 24x20 and 16 actions/frame; LIBERO-long
uses F=4, latent 8x16 and 4 actions/frame. Buffers, RoPE, output unpacking,
first-frame modulation spans, action masking and provisional-cache allocation
use this geometry. The native cache-capacity formula yields 9792 and 2160 tokens
respectively (attention windows 72 and 30). Commit tensor layout is checked
before changing cache state; actual VAE history lengths still require live
integration validation.

`WanVaOperatingPoint` also carries video/action SNR shifts. LIBERO's action shift
is 0.05, versus RoboTwin's 1.0; both use video shift 5.0. These values now determine
both modulation timesteps and Euler sigmas, appear in build declarations and
identity, and mismatched shifts are refused by `assert_point`. Resolved native
configs must supply the shifts explicitly.

CPU tests exercise the full 25V/50A and 20V/50A host sampling loops with a stub
DiT, checking layout, RoPE sizes, pinned first frames, BF16 Euler updates, action
masking, repeated commits and reset. The existing real-checkpoint CPU build
checks for both historical operating points also pass: 11 tests total across
`test_wan_va_checkpoint_geometry.py` and `test_wan_va_engine_operating_point.py`.
These tests do not execute FP8 GPU kernels or establish task quality. VA remains
outside the public FP8 registry pending native T5/VAE/processor integration and
Thor validation of both actual checkpoints.


## VA native conditioning bridge and Thor geometry execution

`instinctflash/runtime/wan_va_engine.py` adds an owned-server bridge. Native reset
still clears streaming VAE state and computes live positive/negative T5 embeddings;
native action processors own normalization and decoding. The bridge validates
schedule, geometry, channel mask and history capacity before replacing the owned
server's transformer. The first observed sample calibrates the DiT without a second
VAE encode or an extra random-noise draw. The real `_ControlLoop` retains deferred
commit of executed actions and observed frames. Unsupported truncated video
schedules, action CFG and all-FP16 recipes are refused. The public backend registry
is unchanged pending full integration qualification.

18 CPU tests pass across `test_wan_va_engine_server.py`,
`test_wan_va_checkpoint_geometry.py` and `test_va_checkpoint_history.py`. They cover
calibration replay, initial and subsequent history windows for F=2/F=4, native
processor delegation, reset and history-position mismatch. These bridge tests use
stub model components and do not certify live T5/VAE processing on Thor.

Actual Thor FP8 eager and CUDA Graph runs now execute LIBERO geometry and the full
20V/50A schedule with action shift 0.05. Both retained two-cycle episodes have
finite outputs, repeat identically and report the expected forward count. The
145 captured graphs produce the same retained action/latent digests as eager.
These are **geometry/kernel checks using RoboTwin weights and random conditioning**;
they do not qualify the LIBERO checkpoint, task success or full Runtime latency.
See [receipts and source hashes](va-geometry-comparison.json).

The RoboTwin 25V/50A FP8 graph regression also passes two-cycle replay and
forward-count checks with its original geometry. This remains a component test
with random conditioning, not the native T5/VAE bridge or a closed-loop result.


## VA RoboTwin live native conditioning validation

The direct `WanVaEngineServer` bridge now runs on Thor with the actual RoboTwin
checkpoint, native BF16 T5, native streaming VAE, real recorded three-camera
observations, and native action pre/postprocessing. Both native conditioning
components reside on GPU. The full 25V/50A schedule and guidance scales 5/1 are
preserved. Native construction uses FSDP elision; matmul TF32, cuDNN TF32 and
cuDNN benchmark are off.

Two three-cycle episodes produce finite 16x2x16 action chunks and repeat exactly
after reset. History positions advance 0 -> 2 -> 4, confirming the live first
commit contains two temporal latents (one initial plus one observed), not the
three-frame synthetic first commit used by old smoke scripts. The follow-up
input-integrity run verifies that changing the camera scene, prompt or executed
action history changes actions, and restoring the original prompt/scene/noise
restores the original action bytes. Both action NPZs were fetched and their hashes
and array assertions independently verified locally.

[Initial receipt](va-native-bridge.json), [input-integrity receipt](va-native-bridge-inputs.json),
[source and artifact provenance](va-native-bridge-provenance.json).

This is direct bridge validation, not public Runtime registration or a closed-loop
success-rate certificate. Recorded frames are supplied independently of predicted
actions; the history control also substitutes recorded executed actions. Timings
in the diagnostic logs are not a matched default-native/FP8 performance comparison.
LIBERO-long still requires its own checkpoint and live conditioning validation.


## VA engine construction and automatic commit

`WanVaEngineLoop` now implements the EngineBackend signature, automatically
retaining the predicted action (or explicit executed-action override) for the
next observed-history commit, matching the default in-process behavior. Reset
clears pending history; close is idempotent and subsequent calls are refused.

`build_native_conditioning` loads the native server in an isolated module namespace
and replaces only its DiT construction hooks with an inert placeholder. Native
T5/VAE construction stays intact. This avoids both loading a discarded native
DiT and inheriting default-path server-class patches. The placeholder refuses
use until the FP8 bridge is installed.

`wan_va_engine_build.py` resolves checkpoint geometry, NFE and guidance from a
copied native config, verifies the Thor device and BF16 conditioning recipe, and
builds the native conditioning plus FP8 loop. Both the isolated-constructor loop
and this factory were tested on Thor for two three-cycle RoboTwin episodes.
All six decoded action chunks equal the earlier full-native-constructor bridge
exactly. 21 CPU tests cover lifecycle, executed-action overrides, constructor
isolation and configuration resolution.

[Loop receipt](va-native-loop.json), [factory receipt](va-factory.json),
[source and artifact hashes](va-factory-provenance.json).

The factory probe supplied the pinned checkpoint directory through a fixed
materialization callback. Public Runtime checkpoint resolution, registry dispatch
and build-declared schedule admission still need integration and verification;
this is not yet public VA FP8 support. LIBERO-long remains unqualified.


## VA public Runtime dispatch (RoboTwin)

`wan_va` is now registered in `ENGINE_BACKBONES`. Its dispatch builds the
checkpoint-specific engine and checks the actual instance's declared schedule
and guidance against the Runtime request, closing a mismatched build before
any prediction. It does not apply the static step-literal check intended for
fixed-schedule frontends. Geometry and SNR-shift checks remain in the native
config/factory/bridge chain. Engine-loop runtime statistics expose the live
build declaration, calibration status, graph count and history position.

The real `Runtime.from_pretrained(local_package, precision="fp8")` path passes
two three-cycle RoboTwin episodes. All six decoded 16x2x16 action chunks equal
the previously validated direct bridge and factory exactly. The original
checkpoint is exposed through a valid declared local package with flat
transformer files and explicit frozen-component reference; no weights changed.
The first attempted package omitted root config/weights and was rejected before
model execution. Its failure receipt/log are retained, and the corrected package
passes normal strict validation.

[Public receipt](va-runtime-registered.json),
[verified action/source artifacts](va-runtime-provenance.json).

CPU dispatch tests also prove that a 2-step built video schedule cannot serve a
25-step Runtime request and that the rejected loop is closed. Existing precision,
geometry, exclusion and operating-point tests pass after updating the former
unsupported-VA fixture to test the unverified-device refusal instead. Public
RoboTwin FP8 execution is established; LIBERO-long, current default-native pairing
and closed-loop task quality remain separate requirements.
