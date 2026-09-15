# VA public Runtime benchmark integration

**Current status:** the LIBERO campaign is complete and fully verified: native
20/20, FP8 18/20 (observed −10 percentage points). See
[the final report](VA_LIBERO_SCREEN.md). Progress counts and “unfinished” statements
below are retained preparation history, not the current LIBERO status. RoboTwin
native startup passed six finite predictions and reset-byte equality; its 40-scene
campaign is complete: clean **19/20**, randomized **20/20**. All 40 episodes
and final source/weight checks passed. The FP8 startup and matched live identity
checks passed. The FP8 campaign and all final checks are now complete: clean
**17/20**, randomized **19/20**, versus native **19/20** and **20/20**.
See [final RoboTwin paired screen](VA_ROBOTWIN_SCREEN.md) and its disclosed
[pre-policy recovery amendment](VA_ROBOTWIN_RECOVERY.md).

The new endpoint `benchmarks.vla.wan_va_runtime_server` accepts either
`--precision native` (default) or `--fp8`. Both execute the public Runtime and
retain the checkpoint's original schedule. The existing legacy VA benchmark
protocol remains separate; its identity cannot admit this endpoint.

`benchmarks.vla.wan_va_runtime_libero_scene` executes one frozen LIBERO scene,
using the native five-zero-action settling sequence, 128-pixel cameras, first
12-action and subsequent 16-action execution windows, and the existing horizon.
It records the complete wire trace and executed actions. Runtime commits the
previous prediction only when the actual subsequent observations arrive. No
repeated frames are substituted for an observed history window.

The benchmark wrapper seeds prediction after deferred history commit using
`episode_seed + native.frame_st_id`. Runtime is constructed without a competing
fixed seed. Resets acknowledge the exact endpoint identity; rejected resets
invalidate the previous episode. Client connections cannot inherit an episode.

## Evidence so far

- 29 local tests passed, covering temporal windows and executed-action parity
  against the legacy protocol, reset/seed behavior, finite-action rejection,
  component-hash and inventory checks, protocol admission, and source discovery
  for the FP8 builder's isolated native module.
- A real LIBERO task-0 / seed-9173 startup fixture contains 29 frames. Independent
  resets produced identical initial camera bytes; NPZ serialization preserved
  the frames. Seed 9173 is excluded by the new evaluation driver.
  [Preparation receipt](va-runtime-libero-startup-preparation.json).
- The first native Thor startup passed six predictions (two three-cycle
  episodes), full 20V/50A, finite 7×4×4 actions, and byte-identical corresponding
  reset outputs. The fetched NPZ and its hash were independently verified.
  [Native startup receipt](va-runtime-libero-native-startup-v1.json).
- The first FP8 attempt stopped during source metadata inspection, before
  startup predictions: `inspect.getfile` cannot locate a class from an isolated
  module absent from `sys.modules`. Source discovery now uses the actual
  constructor's code location; its regression test passes. The failed log is
  retained. The corrected FP8 Thor startup passed six finite 7×4×4 predictions
  and corresponding reset-byte equality, with 180 packed E4M3 tensors and 216
  captured graphs. Its fetched action archive passed independent verification.
  [FP8 startup receipt](va-runtime-libero-fp8-startup-v2.json).
  Native also passed on source-v3, but pairing those receipts exposed a second
  metadata issue: its optimized constructor pointed to the InstinctFlash module
  while the FP8 constructor pointed to the upstream module. The pair was refused.
  [Retained mismatch](va-runtime-startup-comparison-attempt-v1.json).
  Source-v4 now prefers the native class module and uses the constructor only for
  an isolated module; both cases have regression tests. Its native serving
  admission passed, and the independently fetched receipt now records the correct
  upstream fingerprint. No closed-loop episodes used the refused source-v3 pairing.

## Current native closed-loop result

The source-v4 native Runtime completed all 20 fixed LIBERO-10 scenes with
**20/20 successes**. All episode wire traces and executed actions passed both
the runner's verification and independent verification; the final collection
also checked the complete 20-job order, scene/source identities and artifact
hashes. [Native completion](va-runtime-libero-native-complete.json).

The post-run Thor check verified all 364 frozen source/config files, four compiled
libraries, upstream source, startup observations, complete checkpoint component
inventory and hashes, package declaration and unchanged boot identity.
[Final source and weight check](va-runtime-libero-native-source-final.json).

The native service has stopped after verification. FP8 passed serving admission
on source-v4, and both fetched startup action archives passed finite-shape,
reset-byte-equality and hash checks. Pairing verified identical checkpoint,
Runtime/engine/upstream source, packages, startup observations, action-processing
identity, original 20V/50A schedule, geometry, hardware and numerical switches.
[Paired serving admission](va-runtime-libero-paired-serving-admission.json).

FP8 is running the same fixed 20 evaluation scenes. The first eighteen completed
pairs passed independent full wire/action verification and byte-identical
initial-camera pairing: native **18/18**, FP8 **16/18**. Task 2 / seed 20001 reached
the original horizon without FP8 success (796 executed actions after five
settling actions); native succeeded after 242 executed actions. This is a normal
task failure, retained without retry or scene substitution. The next two FP8
scenes, task 3 / seeds 30000 and 30001, both succeeded.
[Incomplete paired progress](va-runtime-libero-paired-progress.json).

These eighteen pairs are an incomplete campaign snapshot, not a final precision-loss
estimate. All 20 FP8 scenes and the final source/weight checks remain required.
The completed native arm alone does not establish population success rate or
real-time reliability.

Startup replays recorded history after predictions; those frames are not the
physical consequences of the predictions. These checks establish neither
closed-loop success rate nor native/FP8 equivalence or real-time reliability.
The current Runtime paired simulator campaign remains unfinished.

## Reproduction and provenance

Run `python -m benchmarks.vla.wan_va_runtime_server --help` in the Thor model
environment. Required inputs include a declared local Runtime package, pinned
model/revision, independently recorded component hash manifest, recorded startup
NPZ, prompt, and a new receipt path. `--startup-only` exits after admission;
otherwise the receipt is published after the socket binds. The endpoint refuses
an ambient `LINGBOT_CKPT` override. FP8 additionally checks packed E4M3 weights,
calibration and captured graphs after startup.

Raw scripts, logs, observations, immutable source archives and action arrays:
`/home/ubuntu/ifl_eval/thor_precision_completion_20260909/va-runtime-campaign/`.
The corresponding Thor directory is under `/home/guanming/ifl_eval/`.
Source-v1 extraction exposed an unresolved worker symlink and was not executed;
source-v2 dereferences it and backs the first native/FP8 attempts. Source-v3
contains the inspection fix and new scene driver: all 364 source/config files
and four compiled libraries were checked during staging. Further production
changes do not retroactively update these frozen measurement sources.

Source-v4 retains the same 364-file scope and four compiled libraries, fixing
native constructor-wrapper source attribution. The simulator runs from its own
verified copy of this frozen source. The existing 20-scene OSMesa manifest's
complete source/assets and simulator-package identity matched the current
environment with `LP_NUM_THREADS=1`. `run_arm.py` records the fixed 20-job order,
retains normal failures, and stops on operational or verification failures.
`verify_episode.py` reconstructs controller actions from complete wire replies,
including the first chunk's skipped latent, and checks reset identity/seeds,
initial observation digest, 12/16-frame windows, horizon and artifact hashes.
A positive synthetic trace and a changed-action negative control both behaved
as expected. All 20 native episodes subsequently passed actual verification;
the FP8 arm and full paired comparison remain required.

## RoboTwin Runtime entrypoint (local validation only)

`python -m benchmarks.vla.wan_va_runtime_robotwin_scene --help` exposes the
observed-history bridge for the original RoboTwin checkpoint. It takes an
existing scene job, frozen scene manifest, explicit Runtime endpoint identity,
output path and simulator/upstream roots. The job contributes the simulator
request; its legacy remote operating point is not used or certified.

The entrypoint requires full 25V/50A, BF16 conditioning, native camera geometry,
six-call startup admission and actual packed weights for FP8. It records both
wire traffic and original-client controller actions, keeps normal task failures,
and refuses to overwrite evidence. The client stages the terminal history
locally without another prediction or remote commit; this differs from the
legacy server's final cache write and is explicitly recorded.

Local tests cover endpoint rejection, original-client bridge/action/history
parity, reset, normal-failure recording and transport cleanup. Real Thor
RoboTwin startup and paired scene verification are still required; this
entrypoint does not yet supply a closed-loop certificate. The running LIBERO
source-v4 snapshot is unchanged.

A real RoboTwin startup fixture is now prepared from `adjust_bottle`, seed 9173,
excluded from all evaluation scenes. Two independent resets each executed 48
initial-pose hold actions, recording cameras every four actions. All 13 frames
per camera repeated byte for byte, and the actual Runtime loader preserved the
1/4/8 history windows after NPZ serialization. Source and asset checks passed
before and after preparation. These are real simulator observations for startup,
not consequences of model predictions or task-quality evidence.
[Preparation receipt](va-runtime-robotwin-startup-preparation.json).

The 40 existing joint-policy frozen scenes were rebound as scene inputs only:
all scene payloads are unchanged, source/assets match the new preparation check,
and the four shared expert preparation/reset functions match the original
preparation snapshot's AST. The raw `robotwin-scenes.json` records the original
manifest hash and protocol. Runtime execution admission remains separate; no
RoboTwin Runtime evaluation has run yet.

All 40 RoboTwin scene requests now pass request/adapter and frozen-scene
validation under the new `source-robotwin-v1` snapshot. This snapshot derives
from LIBERO source-v4, changing only the RoboTwin driver bridge hook and adding
the Runtime bridge and scene entrypoint. Thor staging verified all 366 files
and the four original compiled libraries. This is preparation for execution,
not a completed RoboTwin startup or quality measurement.

The next RoboTwin snapshot, `source-robotwin-v2`, also records the exact initial
pose supplied to native `add_init_pose`, binding it in the episode receipt for
independent controller-action reconstruction. Original arguments pass through
unchanged; a changed origin is refused. All 28 related local checks passed.
Thor staging verified 366 source/config files and four libraries. The prepared
launcher uses native by default and explicitly adds `--fp8` for its other arm;
no RoboTwin model startup or episode has run yet.

The incomplete LIBERO campaign now also includes a normal FP8 failure on
task 7 / seed 70001: 796 executed actions, versus native success after 280.
Both full traces passed independent verification. Neither failed scene was
retried or replaced.

The independent RoboTwin verifier reconstructs controller translations and
quaternion composition from recorded model replies and the actual native origin.
It checks source/scene identity, reset seed acknowledgment, 1/4/8 frame windows,
action geometry, executed prefixes and action digests. In the real simulator
Python environment, reconstruction matched the original client's extracted
pose functions byte for byte. A two-prediction trace fixture passed; changing
an executed action while recomputing both its file hash and action digest was
correctly refused. These are verifier controls, not real policy rollouts.

The RoboTwin batch runner now validates all 40 frozen requests and all 366
source/config files in the real simulator Python environment before execution.
Each episode runs in a fresh simulator process and must pass independent trace
and controller reconstruction before being marked verified. Normal task failures
remain in the ordered results; operational errors or invalid evidence stop the
arm. A separate post-run checker is prepared for the frozen source, compiled
libraries, weights and boot identity. Neither runner completion nor post-run
verification has been claimed before actual execution.

The RoboTwin final comparator requires both complete 40-scene arms and their
post-run source checks, re-verifies every episode, and requires matching initial
cameras/state and controller origins. Clean and randomized counts remain
separate. Controls confirmed that mismatched checkpoint/numerical settings,
changed camera bytes and a missing completed arm are refused. No real RoboTwin
paired result exists yet.

The first real RoboTwin Runtime native episode (`adjust_bottle`, clean seed
50100) succeeded after 111 executed actions and passed independent full
wire-to-controller reconstruction. A verifier metadata bug reused the scene-key
variable for an artifact-hash field; the raw progress label was consequently
wrong, while the raw scene and task result were intact. The original runner,
verifier and inference remain frozen. A corrected verifier revalidated the
complete episode and the collector requires the exact original source hashes
before applying the documented metadata correction. No episode was rerun.
[Amendment](va-runtime-robotwin-verifier-amendment.json).

Independent corrected progress verification now covers both clean
`adjust_bottle` scenes: seeds 50100 and 50101 succeeded after 111 and 119
executed actions. The separate progress report binds each real scene to its
original result and documents the metadata amendment; the running controller
and its original evidence remain unchanged.
[Incomplete native progress](va-runtime-robotwin-native-verified-progress.json).

Timing scope: `RemotePolicy.infer` starts its timer before serialization and
stops it after the synchronous trace append. Recorded `roundtrips` therefore
include request serialization, transport, server work and local evidence writes.
They are diagnostic campaign timings, not isolated model latency or a
real-time certificate. Use a separately controlled latency protocol for speed
comparisons; do not pool these values with the standalone Runtime timings.

Native RoboTwin progress includes a retained normal failure on clean
`blocks_ranking_size`, seed 80100: all 1200 permitted controller actions executed
without task success. Independent reconstruction passed. This scene was not
retried, replaced or used to tune either arm. The incomplete native campaign
now has seven verified episodes, six successful.
