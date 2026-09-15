# Original-weight SDE2 CFG1 cuDNN quality diagnostic

This is the actual BF16 cuDNN numerical variant of the original-weight two-step
cost screen (~443 ms p50), not a trained two-step student. The native quality
bank remains separate at `../two-step-quality`.

The frozen capture plan preserves all 128 historical request identities,
complete SDE `[1, 0.5, 0]`, CFG1, zero padding, trained heads, preprocessing and
external Wan VAE. Attention is explicitly installed on the 28 owned Edge layers
using the existing numerical engine, with frozen vendor source and cuDNN checks.
Capture must observe 7,168 cuDNN full-attention calls and 7,168 native causal
attention calls (128 requests × 2 branches × 28 layers). The frozen two-way
attention source requires the causal subpath to retain its native implementation.

`export_control.py` checks every reference/noise/mask/target byte against both
CFG1 students and the completed native SDE2 bank, including normalizers and
request seeds. It retains native-versus-cuDNN action differences separately.
`score.py COMPACT_DIRECTORY FRESH_REPORT_JSON` checks source and archive hashes,
then independently cross-checks 96 metric vectors and 1,536 bootstrap results
across 16 comparison groups: the existing 15 baselines plus native SDE2.

Prior historical development results were seen; no cuDNN SDE2 quality outcomes
were read before the postprocess plan was frozen. There is no fresh-confirmation,
noninferiority, success-rate or realtime certificate. Raw capture timings are
not latency benchmark measurements.

Capture source on Thor: `two-step-quality-cudnn-v2`; output:
`control-original-sde2-cfg1-cudnn-v2`, under the existing 2026-09-12 study root.
The shared Thor lock is held during capture; H1004–7 ownership is unchanged.
Completion and scoring results must be checked before interpreting quality.

V1 captured all128 requests but failed the incorrect zero-fallback check. Its
source and report are preserved in `failed-v1`; raw arrays remain on Thor. V2
changes only that receipt expectation, not attention kernels or generation.

## Completed result

V2 capture, full input pairing and independent scoring all passed. All128
requests match the native SDE2 bank and both CFG1 students in full input bytes.
The observed native/cuDNN action maxabs is0.0537109375 and meanabs0.0018236187;
these panel values are separate from the shorter latency fixture's deltas.
Independent formulas reproduced96 metric vectors and1536 bootstrap comparisons.

Primary energy-score changes (lower is better; not success-rate percentages):

| Baseline | Joint h1 | Gripper h1 | Joint h32 | Gripper h32 |
|---|---:|---:|---:|---:|
| Same two-step native | +0.295% | +0.082% | -0.0003% | +0.032% |
| Original UniPC4 | +13.41% | +13.91% | +3.03% | -7.30% |
| Original SDE1 CFG1 | -0.47% | -20.57% | +4.16% | -22.82% |
| Trained SDE1 CFG1 cuDNN, two-seed mean | -2.72% | +11.19% | +3.75% | -11.26% |

All four uncorrected95% intervals versus native SDE2 cross zero, which is not
an equivalence certificate. The short joint/gripper intervals versus UniPC4
remain strictly positive. Both gripper intervals versus original SDE1 CFG1
remain strictly negative. See `score_receipt.json` for all16 baseline groups
and intervals; full scores and compact arrays are bound to their local hashes.

The ~443ms attention route has a measured historical quality screen now. Actual
two-step distillation, fresh confirmation and controller/closed-loop admission
remain pending. All V2 capture/postprocess/scoring workers finished successfully.
