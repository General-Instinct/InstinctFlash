# Current pi05 public Runtime × LIBERO paired screen

Both explicit precision arms completed the fixed LIBERO-10 matrix: ten tasks,
two seeds per task, 20 episodes per arm, on Thor.

| Outcome | Native | FP8 |
| --- | ---: | ---: |
| Successes | 16/20 (80%) | 17/20 (85%) |
| Horizon failures | 4 | 3 |
| Success only in this arm | 2 | 3 |

The observed difference is **+5 percentage points** for FP8. Five of twenty pairs
change success outcome, including two native successes that FP8 loses. These
results **do not establish zero loss, superiority or non-inferiority**. Twenty
fixed scene pairs are a screen, not a statistical quality certificate. Whole-runtime
implementation and precision both differ; this does not isolate FP8 arithmetic.

[Paired results and all 20 outcomes](pi05-public-campaign-comparison.json),
[native manifest](pi05-native-campaign-manifest.json),
[FP8 manifest](pi05-fp8-campaign-manifest.json).
All 40 episodes passed wire-packet hash, reset seed/identity, action-array and
observation checks. Every paired initial image/state array is byte-identical.
Each arm keeps the checkpoint's 50-action queue and ten denoise steps. Native
capture passed its six-input startup equality check. No few-step distillation
or action-horizon reduction is part of this comparison.

The simulator pauses for each network reply. Episode wall time and step counts
are not inference latency or real-time deployment qualification. Use the separate
[matched generation latency results](pi05-public-comparison.json) for that workload;
sustained tails and a larger quality study remain unqualified.

## Protocol

- Pinned `lerobot/pi05_libero_finetuned_v044` revision
  `8e174154ef5f6c60a8da12ae99c303d8963138c1`, public `Runtime.from_pretrained`,
  `precision="native"` or `precision="fp8"`, Thor CUDA device 0.
- Two raw 360×360 cameras, native LeRobot `preprocess_observation` and
  `LiberoProcessorStep` on the server. The native conversion flips both image
  axes and forms the 8D state; each runtime owns its checkpoint pre/postprocessors.
- Frozen initial state; ten `[0,0,0,0,0,0,-1]` settling actions, relative controller,
  520-step limit. This differs from the historical VA and pi05 10-action drivers.
- Ten denoise steps, 50-action native queue, one decoded 7D action per simulator
  step. Observation updates do not reset the queue. Reset verifies the declared
  server identity and acknowledges the seed.
- Python, NumPy and Torch seed each request with `episode_seed + action_index`.
  This is a benchmark harness rule, not a public `Runtime(seed=...)` guarantee.
  Calibration/implementation RNG consumption is not isolated; this is a comparison
  of whole runtime routes, not identical-noise quantization arithmetic.

## Retained implementation and setup

The server and smoke scripts retain their measured experiment-specific paths.
The multi-task client below accepts explicit scene, identity and output paths:

- [Server](serve_pi05_runtime_libero.py) requires Thor's pinned LeRobot environment
  plus the measured InstinctFlash overlay. It binds only `127.0.0.1:19051`.
- [Runner](run_pi05_runtime_libero.sh) records each server arm separately. Run it
  under `flock /tmp/thor_gpu.lock`; use one server at a time and stop the owned
  server after the corresponding client completes.
- [Client](probe_pi05_runtime_libero_scene.py) runs in the local pinned LIBERO
  environment, records every request/response and refuses result overwrites.
  Its frozen scene path is
  `/home/ubuntu/ifl_eval/libero_va_20260906/scenes-v2.json`.

Measured raw roots are `/home/ubuntu/ifl_eval/thor_precision_completion_20260909`
and `/home/guanming/ifl_eval/thor_precision_completion_20260909` on Thor. The Thor
source overlay is `pi05-public-sim-source`; Python is
`/home/guanming/frt_env/bin/python`. The local simulator interpreter is
`/home/ubuntu/ifl_eval/libero_va_20260906/sim-env/bin/python`.

Local simulator environment:

```sh
export LIBERO_ROOT=/home/ubuntu/ifl_eval/libero_va_20260906/LIBERO
export LIBERO_CONFIG_PATH=/home/ubuntu/ifl_eval/libero_va_20260906/config
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
export PYTHONPATH=/home/ubuntu/InstinctFlash
```

Forward local port 19051 to Thor's loopback port through SSH. After the server
loads, fetch its identity JSON, then run the client with `--identity PATH`,
`--output NEW_RESULT.json`, `--task 0 --seed 0`. The output includes an NPZ and
`.trace` directory. Match the saved scene and source/asset fingerprints before
comparing with these results. Both measured servers and the SSH tunnel have been
stopped; no background campaign remains from this evaluation.

## Current multi-task campaign

The portable client entry point now expands the fixed scene file into ten tasks,
with seeds `task * 10000` and `task * 10000 + 1`. Each precision arm uses exactly
those 20 initial states. It verifies the frozen simulator source/assets and rejects
an incompatible checkpoint or action horizon before running any episode.

```sh
python -m benchmarks.vla.pi05_runtime_libero_campaign \
  --scenes /path/to/scenes-v2.json \
  --identity /path/to/live-server-identity.json \
  --output /path/to/new-arm-directory \
  --endpoint ws://127.0.0.1:19051
```

Use the pinned simulator environment above. Run each arm against its explicitly
selected server. `campaign.json` freezes the population, source hashes and identity;
each job retains its log, NPZ and wire trace. A failed episode process stops the arm
and records the failure; it does not substitute another seed or overwrite results.
A normal episode that reaches the horizon without success remains a completed
negative outcome. New output directories are required, preventing accidental
mixing of attempts. Both precision arms must finish and their artifacts must be
paired and verified before reporting a comparison. Twenty pairs provide a screen,
not a sufficiently powered non-inferiority certificate.

After **both** arm manifests report completion, verify the entire pair before
publishing the observed outcomes:

```sh
python -m benchmarks.vla.pi05_runtime_libero_compare \
  --native /path/to/native-arm-directory \
  --fp8 /path/to/fp8-arm-directory \
  --output /path/to/new-comparison.json
```

This streams the saved wire packets and checks their hashes, reset identity/seed
acknowledgements, observation shapes, initial observation bytes and action replies.
It also rejects changed populations, mismatched client source/environment identities,
partial runs and incompatible server protocols. The result reports all 20 pairs,
success counts, observed percentage-point difference and discordant pairs; it does
not issue a non-inferiority or real-time certificate. The verifier passed against
the retained native and FP8 smoke traces, and negative tests reject hash-consistent
but semantically changed reset seeds, initial states, actions or mid-episode resets.

## Repeatability and retained attempts

The initial single-scene smoke remains archived separately. Independently started
servers reproduced every retained action from that smoke: native 276 steps and
FP8 257 steps on task 0 / seed 0. This establishes repeatability on that scene,
not equality across precision modes or all inputs.
[Smoke](pi05-public-sim-smoke.json), [native repeat](pi05-native-scene-repeat.json),
[FP8 repeat](pi05-fp8-scene-repeat.json).

The first handoff supervisor stopped after its shutdown identity guard reacted
to the native process exiting. Native termination was confirmed, then a separate
supervisor ran the previously unattempted FP8 arm and completed full pair verification
and shutdown. No episode was replaced or rerun because of that orchestration repair.
Logs and both supervisor scripts remain in the raw root. Both inference servers
and the SSH tunnel have exited.
