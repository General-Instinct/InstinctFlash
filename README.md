<div align="center">

<img src="assets/iFlash.png" alt="InstinctFlash" width="360"/>

**A high-performance serving framework for robotics models.**

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

---

## What's new 🔥

- **Full source and eight model families.** Public install, paired inference and WebSocket serving paths are qualified on Jetson Thor for all eight models below. RTX 5090 coverage is reported separately.
- **Thor — Cosmos3 at full UniPC4/CFG3.** Edge: **1048.01 ms**; Nano: **4772.38 ms**, both native precision with NUMERIC optimizations.
- **Thor — LingBot-VA @2V/4A.** **459.10 ms / 4.51×** versus native 2V/4A in early continuations; full 25V/50A FP8: **2891.74 ms**.
- **Thor — pi05 FP8.** **51.85 ms / 7.88×** versus native, retaining NFE10.
- **RTX 5090 / SM120.** Five model entries now have dedicated paths with real-model evidence. Frozen pi0.5 FP8 also passes cross-process replay and a fresh **500-pair LIBERO Spatial** gate.

## Results

Measurements are hardware- and protocol-specific. Compare reference and candidate **within a
row**; different schedules, reference arms and timing statistics are not a cross-hardware ranking.

### Jetson Thor (SM110)

Prediction p50 on **Jetson Thor** (ms), measured September 15, 2026.

| Model | PyTorch | InstinctFlash |
|:--|--:|--:|
| LingBot-VA | 15506.32 | **2891.74 (5.36×)** · FP8 |
| ↳ LingBot-VA @2V/4A | 2071.29 | **459.10 (4.51×)** · FP8 |
| LingBot-VLA-4B | 624.22 | **221.53 (2.82×)** · FP8 |
| LingBot-VLA-V2-6B | 734.56 | **394.11 (1.86×)** · FP8 |
| Cosmos3 Edge DROID | 3393.78 | **1048.01 (3.24×)** · NUMERIC |
| Cosmos3 Nano DROID | 10184.68 | **4772.38 (2.13×)** · NUMERIC |
| pi05 | 408.58 | **51.85 (7.88×)** · FP8 |
| GR00T N1.7 | 139.50 | **117.30 (1.19×)** · Native |
| DreamZero DROID | 23563.08 | **11899.42 (1.98×)** · FP8 |

VA measures early continuations; its speedups use the native schedule in each row.
Cosmos3 retains full UniPC4/CFG3; detailed execution settings are linked below.
These are speed measurements; task quality is evaluated separately.

[Protocol and raw results](eval/public_release_2026-09-15/results.rst) · [Native VA 2V/4A measurement](eval/va_native_2v4a_2026-09-15/README.rst) · [Reproduction commands](REPRODUCE.rst)

### RTX 5090 (SM120)

Coverage as of **September 16, 2026**: **five verified paths, one partial integration and two
models pending independent SM120 qualification**. “Verified” applies to the checkpoint and
operating point in the linked evidence, not every fine-tune, precision mode or action horizon.
An em dash means no qualified 5090 performance number is published here; it is not a Thor/H100 result.

| Model | SM120 coverage | Reference (ms) | Measured path (ms) | Comparison and evidence |
|:--|:--|--:|--:|:--|
| LingBot-VA (5B) | A1–A7 native chain verified | 389.33 mean | **383.45 mean (1.015×)** | **A7 increment only:** A1–A6 → A1–A7; 42-cycle A–B–B–A, [168/168 actions bit-exact](eval/lingbot_va_robotwin/sm120_wan_qkv_parallel_results.json). |
| LingBot-VLA-4B | Full graph path verified | 220.50 p50 | **97.30 p50 (2.27×)** | Stock → full vision/prefix/action graphs + GPU preprocessing; [6/6 cases bit-exact](examples/lingbot_vla/evidence/reproduce_5090_full_path_results.json). |
| LingBot-VLA-V2-6B | SM120 qualification pending | — | — | [Existing H100/Thor evidence](examples/lingbot_vla_v2/README.md); no independently qualified SM120 path. |
| Cosmos3 Edge DROID | Prompt K/V cache verified | 216.10 p50 | **206.41 p50 (1.047×)** | Native BF16, cross-request cache off → on; [A–B–B–A and six-case exactness gate](eval/cosmos3_edge_5090/persistent_text_kv_results.json). |
| Cosmos3 Nano DROID | Action-only residency verified | 701.26 p50 | **702.36 p50** | CPU-retained unused head → zero-parameter sentinel; [768/768 action values bit-exact](eval/cosmos3_nano_5090/action_only_results.json). **Memory fix, not a speedup.** |
| pi0.5 — LIBERO v0.4.4 | Frozen FP8 path verified | 39.29 p50 | **25.00 p50 (1.57×)** | FlashRT BF16 → frozen FP8; [three-process bit-exact replay and 500-pair gate](examples/pi05_vla/sm120_frozen_results.json). |
| GR00T N1.7 | Partial; SM120 qualification pending | — | — | [BF16/DiT integration exists](examples/groot_n17/README.md), but public full-backbone capture is refused and GPU collate does not default on for SM120. |
| DreamZero DROID | SM120 qualification pending | — | — | [Existing H100/Thor integration](examples/dreamzero/README.md); no independent 5090 performance or task-quality certificate. |

The reference arms are deliberately named: the VA number is **not** the cumulative A1–A7
speedup against stock, and Nano's reference keeps the unused BF16 head on CPU so it fits.
VA reports mean cycle latency; Edge averages per-process p50s across A–B–B–A; pi0.5's FP8
number is the median of three process p50s. Do not combine these into an aggregate speedup.

- **Cosmos scope:** the 5090 evidence uses four steps, **16×8 actions and guidance=1**, not
  the Thor/full-DROID 32-action, CFG3 configuration. Nano retains about **28.46 GiB allocated /
  29.49 GiB reserved** after its gate; these are PyTorch allocator figures, not total process VRAM.
  Text logits and prompt upsampling are outside its action-only path.
- **pi0.5 scope:** the fixed LIBERO checkpoint computes 50 actions and executes 10 per replan,
  with NFE10. Frozen replay is explicitly selected through `FrozenPi05Frontend`; the default
  `load_model()` route is unchanged. See [state persistence, environment requirements and reproduction](examples/pi05_vla/README.md#frozen-execution-state-explicit-opt-in).
- **Verification is not a universal quality claim:** the VLA-4B, VA and Cosmos rows establish
  exactness on their measured inputs/flows. A frontend registration alone is not qualification,
  and those checks are not equivalent to a full closed-loop task-success study.

**Separate task-quality result:** the fresh pi0.5 Spatial run measured **472/500 (94.4%) BF16 →
480/500 (96.0%) frozen FP8**, passing the declared five-percentage-point non-inferiority margin.
The paired difference is **+1.6 points**, with a central 95% interval of **[+0.154, +3.372] points**;
no task collapsed. The conclusion is conditional on this fixed ten-task suite. The linked evidence
also records three identical 33/50 task-5 runs, including the full campaign's task-5 arm.

## Install

```bash
git clone https://github.com/General-Instinct/InstinctFlash && cd InstinctFlash
python3 -m venv .venv-core
source .venv-core/bin/activate
python -m pip install . uv==0.12.5
```

The Python 3.10+ core inspects checkpoints and plans without PyTorch or a GPU.
Inference uses a separate, pinned environment for each model family:

```bash
python3 scripts/bootstrap_vendor.py install pi05 --python python3.12 --root ~/ifl-pi05 \
  --ptxas /usr/local/cuda/bin/ptxas
source ~/ifl-pi05/activate.sh
```

Use `va`, `vla4`, `vla2`, `pi05`, `groot`, `edge`, `nano` or `dreamzero`.
The bootstrap installs the upstream source, compatibility patches, core and adapter.
Model weights are downloaded separately. See [installation and inference](INSTALL.rst)
and the [Thor CUDA backend build](serving/README.rst) for accelerated execution.

## Load a model

**Your fine-tuned checkpoint** — the expected case. Point `serve` at the training output; it
detects the family, writes the small `instinctflash.json` declaration from what the checkpoint
itself proves, and starts serving. One command:

```bash
instinctflash serve /path/to/your/checkpoint
```

Anything the checkpoint cannot prove is asked for explicitly, never guessed. Once the
declaration exists (serve writes it on first run), the same directory also loads in Python:

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("/path/to/your/checkpoint")
```

**A stock release** — use its Hub id after installing the family's environment:

```python
runtime = Runtime.from_pretrained("robbyant/lingbot-va-posttrain-robotwin")
```

| family | model id |
|:--|:--|
| LingBot-VA (5B WAM) | `robbyant/lingbot-va-posttrain-robotwin` |
| LingBot-VLA-4B | `robbyant/lingbot-vla-4b-posttrain-robotwin` |
| LingBot-VLA-V2-6B | `robbyant/lingbot-vla-v2-6b-robotwin` |
| pi0.5 | `lerobot/pi05_base` · `lerobot/pi05_libero_finetuned_v044` |
| GR00T-N1.7-3B | `nvidia/GR00T-N1.7-3B` |
| Cosmos3 policies | `nvidia/Cosmos3-Edge-Policy-DROID` · `nvidia/Cosmos3-Nano-Policy-DROID` |
| DreamZero | `GEAR-Dreams/DreamZero-DROID` |

Fine-tunes reuse their family's adapter; quality is evaluated per checkpoint.

The same `Runtime` defaults to `precision="native"` with a BITEXACT transformation ceiling.
Use `tier_ceiling="numeric"` to allow numerical changes, or `precision="fp8"`
(CLI: `--fp8`) to explicitly enable FP8. Step schedules are selected separately.
See [precision policy](INSTALL.rst#load-and-predict) and
[FP8 support and validation](eval/thor_precision_completion_2026-09-09/README.md).

DreamZero's opt-in [dynamic step cache](INSTALL.rst#load-and-predict) requires
`tier_ceiling="behavioral"` with either precision. See the
[Thor measurements](eval/dynamic_step_cache_integration_2026-09-14/results.md).

## Get actions

**In process** — this is the whole Python API:

```python
with runtime.episode(prompt="put the bottle in the dustbin") as episode:
    while not done:
        result = episode.predict(observation)
        action = result["action"]
```

`observation` is a dict in the model's own format; `result["action"]` contains its action array. No
server to start, no optimization to choose. If a safety layer changed the action before it
reached the robot, pass `executed_action=...` and the model conditions on what actually
happened.

**Over the network** — the `serve` command above hosts the same runtime behind the
msgpack-over-websocket wire protocol the pi0/openpi ecosystem already speaks, so existing
robot-side clients connect unchanged (`pip install openpi-client`):

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy("my-server", 8000)
result = client.infer(observation)
action = result["action"]
```

The prompt rides in the observation; a changed prompt starts a new episode, and a client can
say it explicitly with `{"reset": True, ...}`. Four flags cover the rest:

- `--serve.dry_run` — preflight only: device, declaration, plan. No weights, no GPU.
- `--serve.smoke` — load, produce one action, exit.
- `--serve.seed` — seed native execution for paired comparisons; FP8 serving rejects this option.
- `--serve.viz` — stream observations, actions and latency to a [Rerun](https://rerun.io) viewer.

The second verb, `instinctflash validate <dir>`, checks a checkpoint is publishable; given
`--validate.teacher_outcomes/.student_outcomes/.margin` it also certifies non-inferiority and
stamps the certificate into the package.

## Benchmark acceleration and quantization

After the [vendor and auxiliary-asset preparation](REPRODUCE.rst) and
[native backend installation](serving/README.rst), reproduce a model's paired
eager/default/selected Runtime measurements with the included inputs and fixed
checkpoint revision. Keep the model and asset environments activated:

```bash
python -I -m benchmarks.regression.reproduce prepare --model pi05 --mode fp8 --output pi05-inputs
python -I -m benchmarks.regression.reproduce run --prepared pi05-inputs --output pi05-results
python -I -m benchmarks.regression.serve_smoke --prepared pi05-inputs --output pi05-serving
```

`run` writes checked JSON/CSV reports and full action arrays. `serve_smoke` tests
the actual CLI and WebSocket pipeline across two episodes. Use `--mode native`
for default precision; FP8, numerical compilation and changed schedules are
explicit selections. [Reproduction guide](REPRODUCE.rst).
For the other framework columns, use the [pinned comparison recipes](benchmarks/regression/FRAMEWORK_COMPARISON.rst).

Compare original and optimized models with `instinctflash eval`. Reports separate
latency, action agreement and simulator task success.

```bash
instinctflash eval adapters
instinctflash eval coverage --run /path/to/run
instinctflash eval --registry plan.registry.json report --run /path/to/run
```

See the [evaluation guide](benchmarks/vla/SIMULATOR_EVALUATION.md) to create and run
paired LIBERO / RoboTwin experiments, or [benchmark details](benchmarks/vla/README.md)
for acceleration and quantization protocols. Results: [simulator screening](eval/simulator_quality_2026-09-06/README.md)
and [repeatability, checkpoints and edge latency](eval/simulator_next_steps_2026-09-06/README.md).
The [expanded V2 evaluation](eval/precision_evidence_2026-09-06/README.md) binds
latency and quality evidence to execution profiles and checks explicit control budgets.
The [native qualification workflow](benchmarks/regression/README.md)
adds fresh-start admission, retained failures and checkpoint-specific evidence for each device.
LingBot-VA Hub IDs retain native step counts; 2V/4A requires an explicit `nfe` selection.
The [September 9 Thor comparison](eval/thor_precision_completion_2026-09-09/COMPARISON.md)
separates native acceleration, FP8 Runtime gains and paired task outcomes;
[historical engine controls](eval/fp8_comparison_2026-09-09/README.md) isolate additional implementation effects.

[Shared BF16 fusion](instinctflash/native/bf16/README.md) provides an opt-in NUMERIC path, with per-model compatibility and paired Thor regression results.
[Shared tensor caching and prefill separation](eval/shared_tensor_cache_2026-09-13/README.md) extend native Cosmos optimization to Edge and Nano; exact caching and NUMERIC compilation remain separate options.

## Framework overview

InstinctFlash keeps model declarations, optimization planning, runtime execution, and evidence in
one inspectable path, whether it is called from Python or the command line.

# Architecture

A checkpoint carries a short declaration of what it is. The runtime reads the declaration, decides
which optimizations are provably valid for those weights, applies them, and shows its work:

```
checkpoint ─▶ adapter          ─▶ planner            ─▶ engine passes        ─▶ actions
              declares what        decides what          apply and measure
              the model is         is valid (no GPU,     each optimization
                                   no weights needed)
```

Optimization is organized in six layers, by what each one changes:

| layer | | changes |
|:--|:--|:--|
| 1 | **MODEL** | what is computed — distillation, step reduction, checkpoint compression ([InstinctCompress](https://github.com/General-Instinct/InstinctCompress), [instinct-pdd](https://github.com/General-Instinct/instinct-pdd)) |
| 2 | **GRAPH** | when work is issued — prefill extraction, CUDA-graph capture, memory planning |
| 3 | **CACHE** | what is recomputed — KV reuse, cross-attention and episode caches |
| 4 | **ATTENTION** | how tokens mix — FlashAttention, hybrid and linear attention |
| 5 | **KERNEL** | how a kernel is written — backend and layout dispatch, fusion |
| 6 | **HARDWARE** | what it executes on — fp8/int8, TensorRT, Jetson-class edge devices ([`serving/`](serving/)) |

Layer 1 changes the *weights* and produces a checkpoint; it lives in the companion repos. Layers
2–6 change *how the weights execute* and produce a plan; they are the runtime in this repo. The
layers are not a priority order — the runtime measures where the time actually goes and starts
there.

## Add a model

To add your own model family, declare an `instinctflash.adapters` entry point and `pip install`
your package — see [`examples/external_plugin/`](examples/external_plugin/).

## Roadmap

- [ ] **Few-step distillation, when needed** — only after native optimizations miss a declared
      edge control budget; compare each student with its teacher and the matched untrained
      schedule using paired closed-loop evaluation.
- [x] **LingBot-VA on the edge engine** — native and FP8 serving on Jetson Thor,
      with paired inference and WebSocket checks for full and 2V/4A schedules.
- [ ] **Attention upgrades** — a faster NUMERIC-tier attention arm beside the BITEXACT default
      for pi05-class models; hybrid and linear attention for long-context world models.
- [x] **Cosmos3 and DreamZero on Thor** — pinned installation, paired inference
      and installed CLI/WebSocket checks; task quality remains a separate evaluation.
- [ ] **Device-specific serving defaults** — measure each family and operating point;
      select a verified path within the caller's precision constraints. LingBot-VLA-V2
      native Thor capture and LingBot-VA saturation profiling are complete; selective
      VA action capture showed no speedup and stays experimental.
      [Results and evidence](eval/edge_defaults_2026-09-06/README.md).
      Execution-bound budget selection is available; the expanded V2 H100
      evaluation remains [SCREEN](eval/precision_evidence_2026-09-06/README.md).
