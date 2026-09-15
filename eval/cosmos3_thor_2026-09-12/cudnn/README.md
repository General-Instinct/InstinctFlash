# cuDNN BF16 attention on native Cosmos / Thor

Offline numerical investigation inspired by DreamZero §3.2 and Appendix D.
No production Runtime registration, default change or BITEXACT promotion.
The final action differences require checkpoint-specific quality evaluation;
this experiment is a **SCREEN**, not a declared-margin NUMERIC certificate.

## Why the native path falls back

The pinned upstream orders cuDNN before NATTEN for SM110, but its
`CUDNN_MIN_BACKEND_VERSION = 92200` gate disables cuDNN on the installed 91501
runtime. [The dispatch audit](receipts/cudnn-dispatch-audit.json) records those
source hashes and checks they match the executed screen. Direct PyTorch cuDNN
SDPA works for the actual dense non-causal BF16 GQA shapes tested here. This is
a narrower contract than the complete upstream backend; the evidence does not
justify globally lowering its version requirement.

## Real-operand screen

Capture generation attention operands from the first, middle and last decoder
layers of both CFG branches on the first denoising step. These are actual
checkpoint activations from the recorded DROID fixture, not random tensors.
The native model returns its original attention outputs throughout capture.
An independent native replay must match each captured reference byte-for-byte
before timing its replacement.

| Model | Query shape | KV heads | Native GQA cuDNN speedup over NATTEN | Maximum absolute operator delta |
| --- | --- | ---: | ---: | ---: |
| Edge | 1×3093×16×128 | 8 | 7.63–7.93× | 0.00390625 |
| Nano | 1×3093×32×128 | 8 | 7.64–7.87× | 0.00390625 |

All sampled outputs are finite and all differ in bytes. Differences already
occur at the first sampled generation-attention layer: 0.00048828125 maximum
absolute delta for Edge and 0.0009765625 for Nano. This establishes an operator
implementation difference before any subsequent layers or solver steps can
amplify it; it does not establish which implementation is closer to exact math.

Both direct GQA and explicitly repeated KV heads execute successfully. Direct
GQA is faster and is selected for the full-model candidate. Repeated-head timings
include repetition costs. CUDA Graph timings use three warmups, five operations
per replay and five timed replays. Profiling explicitly confirms
`aten::_scaled_dot_product_cudnn_attention` and a cuDNN SM100-family generated
kernel on this Thor. No alternative SDPA backend is enabled for candidate calls.

The tested environment is Torch 2.10.0+cu130, cuDNN 9.15.1, Thor SM110.
The kernel name includes `f16`, but input and output tensors are BF16; the
full-model wrapper asserts BF16 input/output and retains original output shape.
No inference about dtype is drawn from a kernel name.

## Full-model results

| Model | Native reference p50 (ms) | cuDNN candidate p50 (ms) | Speedup | Action delta max / mean absolute |
| --- | ---: | ---: | ---: | ---: |
| Edge | 2471.14 | 1391.04 | 1.776× | 0.167817 / 0.015321 |
| Nano | 8291.45 | 5546.99 | 1.495× | 0.977512 / 0.085857 |

Both fresh native references reproduce their earlier archived action bytes.
All candidate/reference actions are finite and complete; neither candidate is
byte-identical to its reference. Raw deltas are not task-success percentages.
[Edge comparison](receipts/edge-comparison.json) and
[Nano comparison](receipts/nano-comparison.json) retain p95, per-dimension deltas,
peak allocation and protocol validation. Peak allocated memory is essentially
unchanged (about 10.91 GiB Edge and 35.01 GiB Nano).

## Full-model protocol

The candidate replaces only eligible single-batch dense, non-causal generation
attention. Native causal text attention, masks, token ordering, scale, precision,
checkpoint, four UniPC steps and CFG 3 remain unchanged. The existing native
pointwise/layer-graph optimizations and guarded conditioning cache are enabled
in both arms. The forced backend fails on unsupported execution instead of
silently choosing a different SDPA implementation.

Every arm runs in a fresh process with six warmup and ten measured requests,
two prompts, changing recorded camera frames/states and resets. All 16 complete
32×8 action chunks are retained. Full-model latency includes input preparation
and output handling; operator speedups must not be substituted for request
speedups. Cache admission, reference checks, actual graph replay and absence of
rejection are checked in both arms.

The executed order is Edge candidate, Nano candidate, Edge reference, Nano
reference, serialized with the Thor GPU lock. This is one fresh candidate/reference
pair per model, not a bracketed A/candidate/B performance certificate. The earlier
archived native receipts are additionally checked for identical reference action
bytes and matching protocol/source identity. The Thor clock records September 11;
the directory name identifies this research pass.

The wrapper's numerical override is explicit in the receipts. Its Runtime
construction policy is archived separately as `runtime_declared_execution_policy`;
`execution_policy` labels the actual experimental override SCREEN. Native Runtime
construction does not certify a subsequently monkeypatched attention algorithm.
Python call counters exclude graph replays and are not presented as total GPU
operator counts.

## Reproduction

Use the existing Thor Cosmos environment and cached checkpoints. The inference
code is the same as e09f698; receipts retain exact imported-file hashes, fixture
hash, checkpoint revision, numerical settings and experiment-script hashes.
Put the pinned Cosmos checkout on PYTHONPATH and run from InstinctFlash with the
Cosmos Python interpreter:

```bash
python eval/cosmos3_thor_2026-09-12/cudnn/run.py /absolute/new-screen --mode screen
python eval/cosmos3_thor_2026-09-12/cudnn/run.py /absolute/new-pair --mode pair
python eval/cosmos3_thor_2026-09-12/cudnn/compare.py /absolute/new-pair edge
python eval/cosmos3_thor_2026-09-12/cudnn/compare.py /absolute/new-pair nano
```

Output files must not already exist. The runner holds `/tmp/thor_gpu.lock`,
removes inherited Cosmos optimization flags, and enables the guarded cache.
The comparison checks matching source/protocol/checkpoint, complete finite
actions, archive hashes and cache/graph evidence before reporting latency and
raw action deltas. `--archived-reference PATH.json` additionally compares a prior
matched native receipt's complete actions. No success-rate or acceptable-margin
claim follows from these checks.

Sources: [DreamZero, Appendix D](https://arxiv.org/pdf/2602.15922#page=23),
[PyTorch SDPA backend selection](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).
The local installed version's real execution, rather than current documentation
alone, establishes cuDNN/GQA support for this experiment.
