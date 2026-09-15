# GR00T N1.7

This package adds the `groot_n17` Runtime adapter and a pointer-only local
package for NVIDIA's unmodified `GR00T-N1.7-3B` checkpoint.

The backend keeps NVIDIA's upstream BF16 arithmetic, captures the four repeated
DiT calls, caches prompt/grid-only Qwen3-VL metadata, skips discarded LM logits,
and removes OXE Pose-object decoding overhead. The full stack is gated BITEXACT
against stock under the 6-case protocol (`verify_fastpaths.py`, two prompt-shape
switches, max|delta| = 0.0; results committed in `fastpath_results.json`).

Measured on our H100 (batch 1, NFE=4): stock 114.8 ms p50, DiT capture only
66.8 ms, full stack 59.1 ms. `reproduce_h100.sh` reruns the pair with the exact protocol,
stock eager vs the Runtime DEFAULT arm (its committed run, from the 2026-08-28 re-sweep on a
different H100-80GB box: 94.2 -> 51.8 ms, same 1.8-1.9x class, ours-vs-stock exact equality on
all six gate cases). CPU thread pinning is host-specific (~2 ms here on
a 208-CPU host; large only on very-high-CPU-count hosts) and is therefore
opt-in host configuration, never a package default.

Thor has separate qualification: native DiT capture passes exact startup and
retained-action checks, but increases latency about 7.2% in its paired test and
remains disabled. The earlier VLSA-only DROID FP8 builder shared native fast
decoding and backbone metadata caching. Caching repeated FP32 weight casts further
reduced that recipe to about 127 ms versus 110 ms native, using about 447 MiB of additional
weight storage. Observations and KV remain live; retained FP8 actions match the
prior implementation byte for byte. These timings and the H100
gate above do not certify current FP8 task quality. See the
[current Thor report](../../eval/thor_precision_completion_2026-09-09/COMPARISON.md).

The current Thor recipe additionally quantizes Qwen3 text attention and MLP
projections. Native vision and BF16 DiT remain; the earlier VLSA-only quality
results do not certify this expanded recipe. See the [execution scope](../../eval/thor_fp8_complete_2026-09-10/PROTOCOL.md).

Use `precision="fp8"` in the same Runtime call, or add `--fp8` when serving, to
select FP8 explicitly. Omit it to keep native precision. Current Thor interface
checks cover DROID (40×17 actions), G1 (40×53) and four R1 variants (40×62); these are
input/reset checks, not task-success certificates. Receipts and reproduction are in the
[embodiment report](../../eval/thor_precision_completion_2026-09-09/GROOT_EMBODIMENTS.md).

For another embodiment, set `execution.embodiment_tag` in the checkpoint's
`instinctflash.json`. FP8 reads its weight slot from that checkpoint's
`embodiment_id.json`; both modes use its native modality configuration and
normalization statistics. Pass named `video` and `state` dictionaries using those
modality keys. A tag appearing in the mapping does not imply complete statistics:
the published XDof variants currently have empty wrist statistics and have not
passed this Runtime's inference checks. The DROID-only fast decoder is optional
and does not require DROID configuration in a custom embodiment checkpoint.

```bash
uv pip install -e examples/groot_n17
export GR00T_ROOT=/path/to/Isaac-GR00T                     # upstream checkout
export GR00T_N17_CHECKPOINT=/path/to/GR00T-N1.7-3B         # optional; HF cache works too
# Strict float64/float32/end-to-end exact OXE decoder.
export IFL_GROOT_FAST_DECODE=1
# Cache exact Qwen metadata, prebuild FA2 cu_seqlens, and skip unused logits.
export IFL_GROOT_BACKBONE_FASTPATH=1
```

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained(
    "examples/groot_n17",
    device="cuda:0",
    placement="in_process",
)
runtime.reset(prompt="pick up the object")

# Either pass the official nested GR00T observation, or the compact form:
result = runtime.predict({
    "images": [exterior_rgb, wrist_rgb],
    "state": state_17d,
})
action = result["action"]       # (40, 17)
split = result["actions"]       # eef_9d / gripper_position / joint_position
```

The DiT CUDA Graph is the **default** on qualified capture-capable devices
(Thor retains eager DiT because its measured capture path was slower) — the old
`IFL_GROOT_STATIC_CAPTURE=1` opt-in (a release policy that predates the startup self-check) is
superseded and is now a no-op with a notice (an explicit `=0` is honored as an opt-out). What
makes the default safe is the runtime **self-check**: immediately after each signature's
capture, replay is compared against the upstream eager DiT forward on staged inputs the capture
never saw — every floating-point input redrawn from a dedicated generator, backbone features
included, so a graph that baked any input cannot pass. Exact equality (`atol=0`; the family's
capture tier is BITEXACT). A mismatch releases the graphs, falls back to upstream eager loudly,
and serving continues. Kill-switch (recorded on the plan, printed):

```bash
export IFL_GROOT_NO_CAPTURE=1     # serve eager; IFL_GROOT_SELFCHECK_FAULT=1 drills the FAIL arm
```

GPU collation defaults on for native H100 and Thor, and the explicitly selected FP8 engine, with six live fieldwise byte checks.
`IFL_GROOT_GPU_COLLATE=0` restores upstream collation. This is independent of full
backbone capture, which failed qualification and is refused by the public Runtime.
The full-capture module remains an offline experiment. Native pairs measured 66 → 61 ms on H100 and 135 → 127 ms on Thor.
[Protocol and receipts](../../eval/native_optimization_2026-09-10/README.md).

`fast_decode` and `backbone_fastpath` default to true in this package and can be
disabled with their environment variables set to `0`. `IFL_GROOT_CPU_THREADS` is
opt-in host configuration (unset by default; `auto` caps at min(16, cores)).
The backbone fastpath performs a GPU->CPU sync per forward and must stay off
engine-side CUDA-graph capture paths.

The public FlashRT baseline uses the same upstream policy:

```python
import flash_rt

model = flash_rt.load_model(
    "/path/to/GR00T-N1.7-3B",
    config="groot_n17",
    framework="torch",
    source_root="/path/to/Isaac-GR00T",   # or set GR00T_ROOT
    use_cuda_graph=True,
)
actions = model.predict([exterior_rgb, wrist_rgb], prompt="pick up the object", state=state_17d)
```

`model.action_dict` retains the last split action dictionary. The compact
single-frame image form repeats the current frame to fill the OXE DROID
`[-15, 0]` video history; real evaluation should supply both temporal frames.

Compare the upstream eager control with the same BF16 model using DiT capture:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/groot_n17/benchmark_runtime.py \
  --backend eager --warmup 3 --iterations 20 --nfe 4
CUDA_VISIBLE_DEVICES=0 python examples/groot_n17/benchmark_runtime.py \
  --backend cuda_graph --warmup 3 --iterations 20 --nfe 4
```

Per-fastpath exactness gates: `verify_fast_decode.py` (float64/float32
`np.array_equal` plus end-to-end), `verify_backbone_fastpath.py` (three-case
`np.array_equal` incl. a prompt switch), and `verify_fastpaths.py` (the
committed full-stack 6-case gate). `verify_static_capture.py` gates the DiT
capture alone.
