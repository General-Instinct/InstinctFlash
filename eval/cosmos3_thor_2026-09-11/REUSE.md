# Cosmos3 Thor: reuse before model-specific optimization

Scope: native BF16, four UniPC steps, CFG 3, full 32×8 DROID actions. H100 is
outside this optimization pass. No candidate is promoted by latency alone.

Follow-up measurements: [direct CUDA kernel screen](kernel-reuse/README.md)
and [BF16 Blackwell FMHA adaptation](attention-reuse/README.md). The FMHA
candidate improves compiled Edge/Nano latency but changes action bytes; it is
an experimental numerical screen, not part of the default BITEXACT path.

| Existing implementation | Cosmos adaptation | Admission / current limitation |
| --- | --- | --- |
| GR00T `examples/groot_n17/groot_n17_iwm/full_capture.py`, `StaticFullFlow` shared graph pool | Share a pool across serial decoder captures; reuse `groot_fp8.py`'s owned-output-copy pattern before another graph can overwrite them | Thor primitive tests cover changed inputs, shapes, parameter replacement, interleaved graphs and retained outputs. Both full-model A/B/A comparisons pass; see README.md. |
| Cosmos `nano_action_only.py`, already used on SM120 | Reuse the existing constructor and verifier on Thor; avoid loading the unused untied language head | Native Thor default after full action-byte comparison; explicit opt-out is retained. This primarily saves memory, not arithmetic. |
| VLA-4B/VLA-V2 `instinctflash/runtime/static_tensor_graph.py` | Reuse its bounded capture and eager-fallback design | Its single stateless tensor interface cannot directly wrap Cosmos decoder metadata or mutable caches. Do not silently flatten those inputs away. |
| Existing RMSNorm and RoPE engine kernels | Reuse the fusion strategy while preserving Cosmos's intermediate BF16 casts and native reduction | Existing fused reductions are not automatically BITEXACT. The candidate keeps Torch's mean reduction and checks actual module inputs. |
| Cosmos `persistent_text_kv.py` | Potential cross-request prompt reuse | Existing qualification is guidance=1; the current CFG=3 protocol bypasses it. Extending positive/negative caches requires separate qualification. |
| Engine FP8/FP4 GEMM and mixed-precision paths | Separate optional precision work | Not admitted to this native optimization pass. |

The first per-layer graph prototype used independent pools and was stopped when
Nano's host memory approached capacity and GPU progress stalled. Its incomplete
run supplies no speed or action-equivalence result. Reusing the GR00T pool pattern
initially exposed overwritten retained outputs in the small Thor test; copying
outputs outside the pool fixed that test. This is why pool reuse needs a
Cosmos-specific lifetime adapter even when graph capture itself is already used
in another engine.

After the reuse candidates pass a fresh-process reference/candidate/reference
comparison, profile the remaining request cost before attempting Cosmos-specific
attention, whole-denoiser capture or CFG cache changes. Keep experimental results
separate from README's published speed table until that comparison passes.
