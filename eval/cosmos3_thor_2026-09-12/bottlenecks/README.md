# Native Thor bottleneck investigation

No Runtime changes or speed-table promotion. Profiles use the guarded native
conditioning Runtime from e09f698, BF16, four steps and CFG 3. Scripts are named
for the research pass; the Thor clock/logs recorded September 11. GPU jobs ran
serially under `/tmp/thor_gpu.lock`.

| Model | Unprofiled request ms | Attention share | Matrix/convolution share | Other CUDA events |
| --- | ---: | ---: | ---: | ---: |
| Edge | 2471–2472 | 47.9% | 26.4% | 25.7% |
| Nano | 8287–8292 | 36.6% | 41.0% | 22.5% |

Shares sum individual CUDA event durations; CPU operator totals are not added.
These are diagnostic profiles, not fresh paired latency certificates. Graphs
were warmed before capture and conditioning-cache admission passed. Each run
uses a fixed recorded camera input. The service's independent request RNG
advances: the raw `profile_action_byte_equal=false` compares different requests
and is **not** evidence of a profiler-induced numerical difference. Use the
existing controlled full-action regression for equivalence claims.

The dominant attention kernel is NATTEN's `fmha_64x64x128_sm80_bfloat16`.
Even halving attention time alone would predict only approximately 1.32× Edge
and 1.22× Nano overall GPU-time improvement, holding all other costs constant.
This is an Amdahl estimate, not measured end-to-end speed. Nano also needs large
matrix improvements for a substantial overall gain.

## Same-backend attention tile screen

A synthetic test uses representative Cosmos generation geometries (Q length
3094, BF16, head dimension 128, 16/32 query heads and eight KV heads), two seeds,
and both positive/negative context lengths. These geometries came from the
previous compiled-upstream investigation; the current native Edge trace has
3093 generation rows. CUDA Graph event timings isolate repeated GPU execution.

| Query heads | Default 64×64 ms | 64×128 ms | Output bytes vs default |
| --- | ---: | ---: | --- |
| 16 | 5.27–5.48 | 3.92–4.06 | Different |
| 32 | 10.31–10.55 | 7.65–7.75 | Different |

The 32×128 option also changes bytes. Larger KV tiles improve operator speed,
but change reduction/softmax execution: these candidates are not BITEXACT and
are not promoted. The synthetic check does not quantify task quality.

## DreamZero research connection

The requested [DreamZero paper, §3.2 and Appendix D](https://arxiv.org/pdf/2602.15922)
identifies cuDNN attention and moving UniPC scheduler work onto the GPU as
implementation optimizations. These are the next relevant native candidates.
Its approximate DiT velocity caching, quantization and Flash training need
separate quality qualification; dual-GPU CFG parallelism cannot supply its
reported benefit on a single Thor GPU. Mathematical equivalence of an attention
formula is not sufficient to assign our BITEXACT tier.

[NATTEN's backend documentation](https://natten.org/backends/) confirms that its
general CUTLASS FMHA uses an older kernel family and implements GQA through
repeated tensors. Its hardware-specific backends have different eligibility;
Blackwell branding alone does not establish native SM110 support.

The [cuDNN real-operand and full-model follow-up](../cudnn/README.md) now
measures 1.78× Edge and 1.49× Nano against fresh guarded-native references,
with changed action bytes. It also identifies the upstream cuDNN runtime-version
gate as the reason this installed stack falls back to NATTEN. The candidate stays
an offline numerical screen. The [scheduler-control follow-up](../scheduler/README.md) preserves all tested
action bytes but measures no speed gain from retaining timesteps on the CPU.
It remains an offline rejected candidate; broader packing and mask barriers
are not removed by that experiment.

## Artifacts

`profile-summary.json` contains aggregated CUDA timings; compressed original
profiles retain individual events, source hashes, fixture identity and cache
status. `attention-tiles-20260912.json` retains every tile screen. Full Chrome
traces remain on Thor under
`/home/guanming/ifl_eval/cosmos_runtime_cache_20260911/*-profile-20260912.trace.json`.
Run `summarize.py DIRECTORY` to regenerate the profile summary from the gzip
receipts. `profile_native.py` requires the cached Cosmos environment, repository
and adapter on PYTHONPATH, `IFL_COSMOS3_CONDITIONING_CACHE=1`, the existing DROID
fixture and the Thor GPU lock. `screen_attention.py` requires the same Torch and
NATTEN environment and GPU lock; it does not load a checkpoint.
