# Cosmos3 action policies (Edge / Nano), in InstinctFlash

The current adapter uses NVIDIA's native RoboLab DROID service for camera assembly,
state/gripper conversion, prompt construction and action decoding. Both released
models return **32×8** action chunks with four UniPC steps, CFG 3 and FPS 15.
Edge declares JSON prompts; Nano declares plain prompts. Checkpoint metadata and
the serving declaration must agree. Older exports with empty metadata require
an explicit declaration.

The native and FP8 service bridges share this processing. Native Thor defaults now
use checked pointwise fusion and shared-pool decoder graphs; Nano also reuses
its action-only weight loader. Matched upstream A / candidate / upstream B runs
measured **3,398 → 2,554 ms (1.33×)** for Edge and **10,237 → 8,673 ms (1.18×)**
for Nano, with byte-identical complete actions on all recorded inputs.
[Results and reproduction](../../eval/cosmos3_thor_2026-09-11/README.md) ·
[Automatic regression](../../benchmarks/regression/README.md).

FP8 remains an explicit separate choice. H100 defaults are unchanged. The
historical RoboTwin-service timings below use a different serving contract.

## Run it

```bash
pip install ./examples/cosmos3_policy
# from the cosmos-framework venv with its RoboLab service:
```

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("nvidia/Cosmos3-Edge-Policy-DROID")
observation = {
    "observation/wrist_image_left": wrist_rgb,
    "observation/exterior_image_1_left": exterior_left_rgb,
    "observation/exterior_image_2_left": exterior_right_rgb,
    "state": qpos8,
}
with runtime.episode(prompt="pick up the banana and place it in the bowl") as episode:
    result = episode.predict(observation)
    action = result["action"]  # (32, 8)
```

Use uint8 RGB camera arrays. For the released 540×640 input, `wrist_rgb` has
shape `(360, 640, 3)`. The native service places the wrist view across the top,
resizes each exterior view to `(180, 320, 3)`, and places them side by side below
it. The `image` shortcut accepts this complete `(540, 640, 3)` camera mosaic.
The native Cosmos exterior keys are numbered **1 and 2**. Supply ordinary
instruction text for both checkpoints; the native processor constructs Edge's
JSON prompt and keeps Nano's prompt plain.

For RTX 4090 setup and its explicit NUMERIC recipe, use the
[installation guide](../../INSTALL.rst#rtx-4090) and
[RTX reproduction commands](../../REPRODUCE.rst#rtx-4090) in the activated vendor
and asset environment.

On Thor, `IFL_COSMOS3_CONDITIONING_CACHE=1` enables optional native text-conditioning
reuse through the same Runtime interface. It checks the upstream source and actual
first-use output bytes before cached replay, falls back on rejection, and reports
admission/check/rejection statistics. It uses extra memory and is off by default in BITEXACT mode.
[Runtime measurements and regression](../../eval/cosmos3_thor_2026-09-11/runtime-conditioning/README.md).

For faster native **BF16** on Thor, explicitly permit numerical transformations:

```python
runtime = Runtime.from_pretrained(
    "nvidia/Cosmos3-Edge-Policy-DROID",  # or Cosmos3-Nano-Policy-DROID
    precision="native", tier_ceiling="numeric",
)
```

This selects instance-local cuDNN attention and enables guarded conditioning reuse,
with the same weights, four steps and CFG 3. It targets the screened Cosmos source
and cuDNN 9.15.1; unsupported stacks are rejected. `IFL_COSMOS3_ATTENTION=native`
opts out, and `IFL_COSMOS3_CONDITIONING_CACHE=0` disables reuse. The ordinary
BITEXACT default remains available. Numerical changes are **not** a task-quality
certificate; backend statistics report the selected route and screening status.
Matched Thor p50: **Edge 2469 → 1390 ms (1.78×)** and
**Nano 8305 → 5537 ms (1.50×)** versus our strict path with the same cache enabled.
[Measurements and action differences](../../eval/cosmos3_thor_2026-09-12/runtime-numeric/README.md) ·
[Paired numerical regression](../../benchmarks/regression/README.md#native-numerical-screen).

For explicit FP8, add `precision="fp8"` to the same Runtime call, or `--fp8`
to `instinctflash serve`. On Thor this selects FP8 Q/K/V and dense MLP projections in both MoT towers,
while preserving native vision, attention, output projections, processors and
schedule. H100 retains the Q/K/V recipe. Omit it for native precision. The expanded
Thor recipe requires its own quality evaluation; see the [execution scope](../../eval/thor_fp8_complete_2026-09-10/PROTOCOL.md).

The input `state` contains seven joint positions and one gripper position in the
native RoboLab convention. The service owns gripper conversion. Native RoboLab
observation keys, including the three camera views and state history, are also
accepted. `reset()` restores the request seed stream; no model KV persists across
requests.

## Historical benchmark arms (different serving contract)

| arm | H100 Edge | H100 Nano | note |
|:--|:--|:--|:--|
| NVIDIA stock (robolab, verbatim) | 310.5 ms | 482.3 ms | baseline |
| our pipeline (default) | 235.7 ms | 327.3 ms | optimization knobs OFF |
| + CUDA graphs (`IFL_COSMOS3_CUDA_GRAPHS=1`) | **185.8 ms (1.67x)** | **324.7 ms (1.49x)** | see caveat |

**These are historical results, not current DROID Runtime timings.** The old CUDA-graphs arm uses `torch.compile(mode=
"reduce-overhead")` over the same weights, and inductor's cudagraph_trees **asserts on a prompt
change** — the speedup holds for single-prompt workloads only; multi-prompt serving must stay on
the pipeline arm. On Jetson Thor the graphs arm is measured *slower* (676 vs 660 ms Edge):
capture pays on launch-bound GPUs and this is not one. Tier: NUMERIC (vs our own eager:
<=1.6e-2 Edge / <=5e-2 Nano; null controls 0.0). Matched-input actions differ systematically
from the NVIDIA server (input-assembly/RNG alignment is an open task) — the latency comparison
stands; an equivalence claim does not.

## Reproduce the README H100 rows

```bash
IFL_COSMOS3_MODEL=edge CUDA_VISIBLE_DEVICES=<idle-gpu> examples/cosmos3_policy/reproduce_h100.sh
IFL_COSMOS3_MODEL=nano CUDA_VISIBLE_DEVICES=<idle-gpu> examples/cosmos3_policy/reproduce_h100.sh
```

Three arms, strictly serialized, byte-identical measure clients per arm.

## Attribution

Cosmos3 and its DROID policy checkpoints are NVIDIA's (OpenMDW-1.1). Nothing is vendored here
beyond two small measure clients and a guardrail-no-op launcher; the adapter imports the
patched checkout in-process.


The following author measurements use the pinned 16-action, guidance-1 SM120 protocol;
they do not replace the current DROID checkpoint configuration or its Thor measurements.

## RTX 5090 persistent prompt K/V

Cosmos already proves that text K/V are independent of the changing action and vision tokens:
it reuses them from denoise step 1 across steps 2–4. Upstream discards that cache at the end of
every request. On RTX 5090, InstinctFlash keeps the completed cache in a transactional,
prompt-keyed four-entry LRU and therefore skips the same text prefix on the next control cycle.
New images and states reuse it; a changed transformed prompt gets a separate cache.

Fresh-process A-B-B-A on the official Edge checkpoint measured **216.10 → 206.41 ms**
(**1.0469x**, -9.69 ms), with both baseline/candidate arm spreads below 0.77%. All six action
cases were exactly equal (`max|delta| = 0`), including four changed observations and a prompt
switch. Failed requests never publish partially initialized per-layer caches. Guidance other
than 1.0, negative/upsampled prompts, velocity postprocessing, sharded execution, temporal-causal
video, and sound bypass or refuse the optimization.

The cache defaults on only for SM120 and can be disabled with
`IFL_COSMOS3_PROMPT_KV_CACHE=0`. Other architectures retain their previous default unless the
operator explicitly opts in. Reproduce with
`eval/cosmos3_edge_5090/benchmark_persistent_text_kv.py`; the ABBA evidence and rejected graph
ablations are in `eval/cosmos3_edge_5090/persistent_text_kv_results.json`.

## RTX 5090 Nano action-only residency

Cosmos3-Nano's Qwen wrapper contains an untied `(151936, 4096)` BF16 `lm_head` even though
action serving sets `predict_text_tokens=False` and never executes it. On a 32 GiB RTX 5090,
materializing that dead 1.159 GiB head is the difference between load/request OOM and a complete
four-step inference. The SM120 Nano path consumes a one-shot constructor token while the module
is still on `meta`, verifies the exact head shape and that it is not tied to the input embedding,
and installs a zero-parameter sentinel before `to_empty(cuda)`.

The real Nano checkpoint completed inference at `(16, 8)` with 28.46 GiB allocated / 29.49 GiB
reserved after the six-case gate. A full-checkpoint A/B keeps the original BF16 `lm_head.weight`
on CPU in the reference arm and uses the sentinel in the candidate; all 768 float32 action values
match bit-for-bit (`max_abs=0`). The candidate's four-step p50 was 702.36 ms versus 701.26 ms for
the CPU-head reference, so this is a residency fix rather than a claimed kernel speedup. Reproduce
with `eval/cosmos3_nano_5090/verify_action_only_residency.py`; the exact hashes, protocol, and
memory evidence are in `eval/cosmos3_nano_5090/action_only_results.json`.
Text logits and prompt upsampling fail loudly instead of silently running without their head.
Disable the action-only path with `IFL_COSMOS3_NANO_ACTION_ONLY=0`; other architectures and the
Edge checkpoint retain their previous residency.
