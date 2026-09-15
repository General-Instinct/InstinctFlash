# VLA 4B/V2: current Thor native/FP8 RoboTwin evaluation

Both VLA 4B and V2 completed all 40 paired scenes per model (160 total
rollouts). All endpoint startup admissions passed on real held-out RoboTwin
observations. These are small paired screens, not non-inferiority or real-time
certificates; startup admission does not prove all-input BITEXACT equivalence.

| VLA V2 setting | Native | FP8 | Observed difference | Native-only / FP8-only |
| --- | ---: | ---: | ---: | ---: |
| Clean | 17/20 | 18/20 | +5 percentage points | 1 / 2 |
| Randomized | 16/20 | 18/20 | +10 percentage points | 0 / 2 |

V2 clean has sixteen both-success pairs, one native-only success
(`blocks_ranking_size`, seed 80100), two FP8-only successes (`click_alarmclock`,
seed 90100; `click_bell`, seed 100101), and one both-failure pair. Randomized has
sixteen both-success pairs, two FP8-only successes (`beat_block_hammer`, seed
70101; `blocks_ranking_rgb`, seed 80100), and two both-failure pairs. Aggregate
counts do not establish superiority or a universal quantization-loss estimate.

All 80 V2 results and wire/controller traces passed verification. Each pair has
byte-identical initial camera/state arrays and matching scene, checkpoint,
source, package and schedule identities. The final Thor check verified 385
source/config files, four compiled libraries and the startup input unchanged.
All failures remain in the records; no task was retried or replaced.
[Complete V2 pairs](vla2-joint-paired-comparison.json) /
[native completion](vla2-joint-native-complete.json) /
[FP8 completion](vla2-joint-fp8-complete.json) /
[final source check](vla2-joint-fp8-source-final.json).

| VLA 4B setting | Native | FP8 | Observed difference | Native-only / FP8-only |
| --- | ---: | ---: | ---: | ---: |
| Clean | 15/20 | 14/20 | −5 percentage points | 2 / 1 |
| Randomized | 15/20 | 16/20 | +5 percentage points | 3 / 4 |

All 80 results and wire/controller traces passed verification. Each pair has
byte-identical initial camera/state arrays and matching scene, checkpoint,
source, package and schedule identities. The final Thor check verified all 385
source/config files, four compiled libraries and the startup observation unchanged.
All normal task failures remain in the report; no task was retried or replaced.

All four `handover_block` scenes (two seeds in each setting) succeeded with
native precision and failed with FP8. This task-specific pattern warrants
follow-up; aggregate success counts must not obscure it. These are small screens,
not population-loss estimates, non-inferiority or real-time certificates.

[Complete paired results](vla4-joint-paired-comparison.json) /
[native completion](vla4-joint-native-complete.json) /
[FP8 completion](vla4-joint-fp8-complete.json) /
[final source verification](vla4-joint-fp8-source-final.json).

| Checkpoint | Pinned revision | Decoded action chunk |
| --- | --- | --- |
| `robbyant/lingbot-vla-4b-posttrain-robotwin` | `fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e` | 25 × 14 |
| `robbyant/lingbot-vla-v2-6b-robotwin` | `0451855729ec904f970600e0aec8b84661423afe` | 50 × 14 |

Both precision arms retain ten action-denoising steps, native observation/action
processing, three live cameras, and native joint-position control. Native uses
the admitted capture path; VLA 4B requests a BITEXACT tier ceiling, V2 a NUMERIC
ceiling. These ceiling names are admission policies, not universal equivalence
proofs. FP8 explicitly selects the engine and retains BF16 vision.

The measured startup numerical environment has matmul TF32 enabled in native and
disabled in FP8. Both arms have cuDNN TF32 enabled, cuDNN benchmark disabled, and
deterministic algorithms disabled. The comparison therefore measures complete
implementation choices, not an isolated change to FP8 arithmetic.

## Verified startup evidence

Eight calls per endpoint used the same three uint8 RGB 240 × 320 cameras and
14-dimensional float64 state, preserving input bytes. The FP8 endpoints verified
72 E4M3 weight tensors, native BF16 vision parameters, and eight graph replays.
The native endpoints recorded 68 denoising replays; V2 also recorded five vision
and five prefill replays. Checkpoint, upstream, Runtime, adapter, pipeline,
hardware and package identities match between precision arms for each model.

- [VLA 4B native](vla4-joint-native-startup.json) / [FP8](vla4-joint-fp8-startup.json)
- [VLA V2 native](vla2-joint-native-startup.json) / [FP8](vla2-joint-fp8-startup.json)
- [Real startup-observation preparation](vla-joint-startup-preparation.json)

The startup scene is clean `adjust_bottle`, seed 9173. Its expert trajectory
passed; independent native resets and NPZ serialization preserved every input
array byte. It is excluded from evaluation. Its observation SHA256 is
`f681d40e2ff4d0695807597fa6c8c80911fde97835a16540b903d834527d8de9`.

## Paired campaign protocol

Each model has ten tasks × two seeds × two settings × two precision arms:
80 rollouts. Clean and randomized each contribute 20 pairs and are reported
separately. Existing frozen scenes were checked against current simulator source,
asset hashes, and the `fresh-clutter-bounds-v1` reset policy. Their selection
preceded this FP8 campaign. No failed task is retried or replaced.

RoboTwin pauses while waiting for actions. The driver applies each generated
joint action until success or the native task horizon; it does not shorten task
horizons or replace native controllers. This is a small SCREEN, below the
declared 100-pair gate per setting, not a non-inferiority or real-time certificate.

Each rollout retains its request, process exit status, result, and complete
hashed wire trace. Incremental verification reconstructs the executed action
prefix from model replies and compares it exactly to the controller action
trace and digest. It also checks reset acknowledgements, finite native shapes,
frozen scenes, checkpoint identity, and simulator source/assets. Final pairing
must additionally verify identical initial camera/state bytes across precisions.

## Reproduction entry points

The repository endpoint is `python -m benchmarks.vla.joint_policy_server`.
Use `--mode runtime_default --precision native` for native, or `--fp8` for FP8.
Both take pinned `--model` and `--revision`, `--startup-observation`,
`--startup-prompt`, `--port`, and a new `--receipt` path. `--startup-only` checks
admission without opening a listener. Native capture qualification additionally
uses `--require-capture --tier-ceiling bitexact` (4B) or `numeric` (V2).
FP8 refuses the bitexact ceiling and native capture/noise flags.

Prepare a real calibration observation with
`python -m benchmarks.vla.prepare_joint_startup --help`. Run bound rollouts with
`python -m benchmarks.vla.joint_robotwin_driver --request JOB --output RESULT`.
Simulator roots, checkpoint environments and a live identity-bound endpoint are
required; a draft plan alone is not executable.

Campaign-specific scripts, manifests, observations and results are retained at
`/home/ubuntu/ifl_eval/thor_precision_completion_20260909/vla-joint-campaign/`.
The Thor source snapshot is in the matching directory under `/home/guanming/`.
It contains 385 verified source/config files plus four hashed compiled libraries;
archive SHA256 is `0df3e30f5c2b12df1ecbc55cc49255285912be658e1ce06535ee34966e875a27`.
These absolute paths describe the retained experiment, not a portable installation.
