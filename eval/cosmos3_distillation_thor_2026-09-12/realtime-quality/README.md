# Thor V16 deployed quality capture

This study preserves the frozen Compress historical-development protocol:
16 observations, eight explicit request seeds each, all four final-online64
Edge SDE1 students and both native BF16/cuDNN variants. The prospective local
plan is `plan.json`; native producer protocol bytes are copied unchanged as
`protocol.json`. These are development screens, not fresh confirmation or
closed-loop success measurements.

`run_capture.py CHECKPOINT FRESH_OUTPUT --data DATA --configuration ID
--attention native|cudnn` records all 128 requests under the shared Thor lock.
It validates protocol/data/package/source identities, the actual external VAE
load and the declared one-step guidance. Put the frozen realtime-deployment
script directory on PYTHONPATH together with the normal Flash/Compress paths.

`capture_request.py` observes the installed Runtime and sampler, using explicit
request seeds. Each request saves raw/model actions and command targets, full
endpoint, effective sampler reference/mask and initial noise. Normalizer metadata,
actual seed, timestep, preparation/branch counts and restored bindings accompany
the arrays. The original preparation-mask trace can precede the padding wrapper;
the separate saved `preserve_mask` is the actual mask passed to the sampler.

The producer capture helper is an unchanged copy whose hash is bound by the
producer's protocol. Two CPU integration tests passed against the actual native
sampler with a tiny model and fixture normalizer. GPU visibility was disabled;
CUDA RNG isolation was replaced with CPU RNG isolation only for those tests.
The complete framework venv is required (the lighter test venv lacks loguru).
These tests do not establish full-model quality or native/H100 parity.

No quality conclusions may be drawn until the producer completion and independent
audit succeed and all deployed captures and scores have been validated. The
latencies of instrumented quality calls are not published benchmark timings.

`validate_capture.py DIRECTORY` verifies every request hash, full finite arrays,
actual seed/callback/branch counts, preserved context, zero padding and normalized
action slice. Its result is execution validation, not a policy-quality certificate.
`run_all.py STUDY_ROOT --capture-source CAPTURE_SOURCE --status FRESH_STATUS`
serializes remaining variants, validating any existing complete bank before reuse.
An existing incomplete bank stops the queue for investigation. The status records
the child PID and command so a later continuation can check the actual process.

The frozen capture runs from `realtime-quality-v1` on Thor, alongside
`realtime-deployment-v2`; the orchestration code is in `quality-orchestration-v1`.
The live queue receipt is
`/home/guanming/ifl_eval/cosmos_distill_20260912/quality-panel-live-v1.json`.
First CFG1/seed12031 native and cuDNN banks completed 128 requests each. The native
bank passed full-array validation. Full fields remain outside Git under the
Thor study root and `/home/ubuntu/ifl_cosmos_quality_preflight/` after transfer.

A first-request H100/Thor check matched request seed, normalizer, effective mask
and both command targets exactly. Full conditioning-reference and initial-noise
hashes differed; raw/model action maxabs difference was 0.015625. The generator
mixes preserved visual context into initial noise, so this alone does not locate
the discrepancy in random versus conditioned coordinates. No cross-device
bitexact claim is made; that distinction needs additional input-field evidence.

`export_pair.py NATIVE_BANK CUDNN_BANK FRESH_OUTPUT --source CAPTURE_SOURCE`
first validates both full banks, then requires identical request identities,
normalizers, sampler arguments and byte-identical effective references, masks,
initial noises and command targets. It exports only the four action/target arrays
in protocol order, with hashes linking the compact banks back to all full captures.
It refuses mismatched pairs or existing outputs. Four CPU tests cover successful
export and rejection of changed seeds, normalizers and initial noise.

The actual CFG1/seed12031 pair passed all 128 requests. Raw and model action
maxabs between native and cuDNN was 0.03125. This is a numerical screen, not a
quality score. Compact arrays and the full pairing report are retained outside Git
at `/home/ubuntu/ifl_cosmos_quality_preflight/quality-pair-cfg1-seed12031-v1`.

Both CFG1 seeds passed pairing with maxabs 0.03125 each. CFG4/seed12031 also
passed, with maxabs 0.21875. These deltas cover 128 historical requests per pair;
they are separate from the shorter latency-fixture deltas.

The frozen producer analysis and independent auditor were replayed on CPU into
`/home/ubuntu/ifl_cosmos_quality_preflight/producer_cpu_replay_v1`, leaving the
producer's outputs unchanged. Both exited zero; the audit checked 3,547 files,
10 configurations, 1,280 requests and 2,112 bootstrap comparisons. The commands,
result hashes and scope are recorded in `producer_replay_receipt.json`. This
replay does not replace the producer's final completion inventory or validate
the Thor action-changing variants.

## Completed deployed numerical screen

All eight variants completed full capture validation (1,024 requests). All four
native/cuDNN pairs passed identical effective-input and target checks. The CFG4
pair's maxabs is 0.21875 for both seeds. Raw fields remain outside Git.

`score_pairs.py PAIR_ROOT CPU_REPLAY_DIRECTORY FRESH_REPORT` loads all four
compact pairs, the frozen teacher bank and the independently audited H100
analysis. It requires both training seeds and both attention paths. It reuses
the frozen producer scores, then cross-checks 768 raw variant metric vectors
and 1,344 bootstrap comparisons with the separately implemented auditor formulas.
The complete real panel passed. Two-seed means average within 16 episodes;
they do not increase the number of independent episodes.

The full report is
`/home/ubuntu/ifl_cosmos_quality_preflight/deployed_scores_v1.json`.
`deployed_score_receipt.json` retains report hashes, all four primary metrics
and their paired bootstrap intervals for every comparison. The table below
shows percentage changes in those **energy scores**, not task success rates.
Lower scores are better; values are means across the two training runs.

| Comparison | Joint h1 | Gripper h1 | Joint h32 | Gripper h32 |
|---|---:|---:|---:|---:|
| CFG1 cuDNN vs same student native | +0.53% | −0.19% | −0.09% | +0.01% |
| CFG4 cuDNN vs same student native | +1.82% | +1.67% | +0.05% | +0.28% |
| CFG1 cuDNN vs original UniPC4 | +13.44% | +2.76% | −1.51% | +4.91% |
| CFG4 cuDNN vs original UniPC4 | +6.84% | −5.22% | +0.66% | +18.49% |

The original-control comparisons are Thor versus H100 historical outputs, not
cross-device bitexact inputs. Their uncorrected intervals show a positive
short-horizon joint-score delta for CFG1 cuDNN and a positive long-horizon
gripper-score delta for CFG4 cuDNN. Most other intervals cross zero; that does
not establish equivalence. All metrics and seeds remain in the full report;
there is no aggregate winner score or retrospective noninferiority margin.

These are diagnostics on reused development observations. Producer completion
inventory, fresh confirmation, executed controller chunk/deadline and closed-loop
task success remain pending. The numerical cross-check does not replace those
gates or certify a realtime policy.
