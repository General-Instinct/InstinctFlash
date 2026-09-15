# GR00T Thor embodiment interface checks

Pinned checkpoint: `nvidia/GR00T-N1.7-3B` revision
`2fc962b973bccdd5d8ce4f67cc63b264d6886495`. Six embodiments pass public
`Runtime.from_pretrained(..., placement="in_process", precision=...)` checks
in both native and FP8 modes. These are interface checks, not task-success or
latency certificates.

| Embodiment | Cameras | Action shape | Calls per precision | Receipt |
| --- | ---: | --- | ---: | --- |
| DROID | 2 | 40×17 | 7 | [Pair](groot-embodiment-droid-pair.json) |
| G1 | 1 | 40×53 | 6 | [Pair](groot-embodiment-real_g1_relative_eef_relative_joints-pair.json) |
| R1 base | 3 | 40×62 | 8 | [Pair](groot-embodiment-real_r1_pro_sharpa_relative_eef-pair.json) |
| R1 Mecka | 1 | 40×62 | 6 | [Pair](groot-embodiment-real_r1_pro_sharpa_relative_eef_mecka-pair.json) |
| R1 MaxInsights | 1 | 40×62 | 6 | [Pair](groot-embodiment-real_r1_pro_sharpa_relative_eef_maxinsights-pair.json) |
| R1 Human | 3 | 40×62 | 8 | [Pair](groot-embodiment-real_r1_pro_sharpa_relative_eef_human-pair.json) |

Every pair retains the checkpoint's four action steps, processors and modality
keys. The probe independently changes each recorded camera, synthetic state and
per-call prompt. Each change affects the action. Repeated inputs and reset restore
identical action bytes within each precision. FP8 checks verify E4M3 projection
weights, executed VLSA replays and the checkpoint-declared weight slot. Native
parameters contain no float8 tensors. The 82 retained action chunks were checked
locally for artifact hashes, dimensions, finiteness and these input/reset results.
Public handles were closed, the final campaign exited successfully, and no probe
or GPU compute process remained after completion.

The two XDof variants remain unqualified. Their published wrist state and
absolute-action statistics are empty despite required state conditioning.
XDof subtask failed in native decoding before the FP8 arm. Relative-action
statistics cannot supply the missing state normalization. No replacement
statistics were invented. [Pinned file audit](groot-statistics-audit.json).

An upstream follow-up found the current public base revision unchanged and the
current `state_action_processor.py` byte-identical to the measured installation
(SHA256 `0151dd0bb6cb727fd401bf78121efc532e7184f2575e6ef25294085f629ac145`).
This check did not find a supplied fix for the empty XDof statistics.

For paired simulator evaluation, the official LIBERO-10 fine-tune has been
downloaded at revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`.
Its model predicts 40 steps but native action decoding emits 16; both precision
arms must preserve that distinction. Seven inference files totaling about 6.91 GB
and all 1031 BF16 indexed tensors have been checked. This is preparation only;
native/FP8 inference and task evaluation remain pending.
[Checkpoint receipt](groot-libero10-checkpoint-preparation.json) /
[official checkpoint](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO/tree/2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21/libero_10).

The existing `benchmarks.vla.groot_policy_server` now accepts `runtime_fp8`
alongside `stock` and `runtime_default`. It uses the same native LIBERO driver,
16×7 decoded chunk and eight-action execution horizon. FP8 startup requires an
explicit `--startup-observation` NPZ rather than implicitly calibrating on the
old all-zero shape probe; the input hash and startup prompt enter endpoint
identity. The NPZ must contain native flat `video.image`, `video.wrist_image`
(256×256×3 uint8) and `state.x/y/z/roll/pitch/yaw` (float32 or float64 length one), plus
`state.gripper` (float32 or float64 length two). Use the same declared startup observation
and prompt in both arms. The engine's sequence-shape-dependent VLSA calibration
remains part of the measured implementation, not a claim of isolated arithmetic.
The endpoint verifies packed E4M3 Q weights and an executed VLSA replay before
advertising FP8. Sixteen server/driver tests pass, and both Thor endpoints completed their
20-scene paired screen. [Results and protocol](GROOT_LIBERO_SCREEN.md).

The native simulator startup fixture has now been generated with task 0, seed
9173, reserved for calibration and excluded from evaluation. Two native resets
and the NPZ roundtrip preserve observation bytes, including native float64 state;
the existing model boundary performs its normal float32 conversion. The driver
now rejects a declared LIBERO checkout that differs from the imported benchmark
code or configured BDDL, initial-state and asset directories. It also checks the actually imported GR00T
simulator wrapper, including a module cached from another checkout. A real
environment recheck matches the prepared fixture’s recorded source hashes.
[Source-binding verification](groot-libero-source-binding.json).
[Preparation receipt](groot-libero-startup-preparation.json).
The qualified raw fixture is `groot-libero-startup-9173-v3/observation.npz` under
the local campaign root. Earlier attempts remain: the first rejected native
float64 state; the second completed without checking actual source binding and
is not used as qualified provenance. The third passed the source-binding check.
This prepares input only; it does not establish Thor FP8 task quality.

The completed native/FP8 LIBERO-10 screen has 20 frozen initial scenes (ten tasks,
two seeds each), covering all 40 episodes. All requested scene keys and
source identities were verified against the draft; calibration task 0 / seed
9173 is excluded. The native Thor endpoint now passes its real-observation startup probe with
finite 16×7 output, matching the declared checkpoint, precision and calibration
identity. Its 20-scene native campaign completed with 20 successes; all action traces,
fixed scene identities and endpoint provenance passed verification. All 371
frozen source files and four compiled libraries were rechecked unchanged.
[Native completion manifest](groot-libero-native-complete.json).
FP8 startup also passes with finite 16×7 output, four packed E4M3 Q weights
and an executed VLSA replay. Its checkpoint, pipeline, startup input, prompt
and numeric environment match the native arm. The FP8 campaign completed with 17/20 successes versus native 20/20,
three native-only successes and no FP8-only successes. The observed −15-point
difference is a small-screen result, not a population loss estimate.
[Full paired report](GROOT_LIBERO_SCREEN.md).
[FP8 startup receipt](groot-libero-fp8-startup.json). [Native startup receipt](groot-libero-native-startup.json).
This screen does
not satisfy the declared 100-pair minimum for a non-inferiority certificate.
[Preparation](groot-libero-campaign-preparation.json) /
[frozen scenes](groot-libero-campaign-scenes.json).

## Reproduction and retained failures

Use the model's upstream GR00T environment, Thor kernels, and `GR00T_ROOT` from
the family setup. Run each precision separately with a new output directory:

```bash
python eval/thor_precision_completion_2026-09-09/probe_groot_embodiment.py \
  --checkpoint /path/to/pinned/GR00T-N1.7-3B \
  --frames /path/to/calib_obs.npz \
  --embodiment real_r1_pro_sharpa_relative_eef \
  --precision fp8 --output /path/to/new-result
```

The probe creates a symlink package with the requested embodiment declaration;
weights remain unchanged. Source hashes are in each receipt. Recorded frames
come from `/home/guanming/ifl/t3_assets/calib_obs.npz` on the measured host.
Raw results are under
`/home/guanming/ifl_eval/thor_precision_completion_20260909/embodiment-v2-*`
(DROID/G1) and `embodiment-v3-*` (R1). R1 v3 initializes valid reference rotations
from declared action formats. Earlier failures remain: a probe accessed the lazy
native policy before reset, and an R1 probe supplied an invalid zero rotation by
relying on field-name suffixes. Corrected attempts use separate directories.

The measured `groot-embodiment-source` overlay includes the checkpoint slot-mapping
fix. The later local guard that skips DROID decoding optimization for checkpoints
without DROID configuration has unit coverage, but is not part of these source
receipts. The pinned checkpoint includes DROID configuration in all six runs.
