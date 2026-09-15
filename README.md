<div align="center">

<img src="assets/iFlash.png" alt="InstinctFlash" width="360"/>

**A high-performance serving framework for robotics models.**

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

---

## What's new 🔥

- **Full source and eight model families.** Public install, paired inference and WebSocket serving paths are qualified for all eight models below.
- **Cosmos3 at full UniPC4/CFG3.** Edge: **1048.01 ms**; Nano: **4772.38 ms**, both native precision with NUMERIC optimizations.
- **LingBot-VA @2V/4A.** **459.10 ms / 33.78×** versus the full 25V/50A native reference in early continuations; full 25V/50A FP8: **2891.74 ms**.
- **pi05 FP8.** **51.85 ms / 7.88×** versus native, retaining NFE10.

## Results

Prediction p50 on **Jetson Thor** (ms), measured September 15, 2026.

| Model | Native PyTorch | LeRobot | vLLM-Omni | InstinctFlash |
| --- | ---: | ---: | ---: | ---: |
| LingBot-VA | 15506.32 · 25V/50A | — | Unsupported | **2891.74** · FP8, 25V/50A |
| ↳ LingBot-VA @2V/4A | — | 1171.08 · native, 2V/4A | Unsupported | **459.10** · FP8, 2V/4A |
| LingBot-VLA-4B | 624.22 · NFE10 | Unsupported | Unsupported | **221.53** · FP8, NFE10 |
| LingBot-VLA-V2-6B | 734.56 · NFE10 | Unsupported | Unsupported | **394.11** · FP8, NFE10 |
| Cosmos3 Edge DROID | 3393.78 · UniPC4 CFG3 | Unsupported | 1074.11 · compiled, UniPC4 CFG3 | **1048.01** · BF16 NUMERIC, UniPC4 CFG3 |
| Cosmos3 Nano DROID | 10184.68 · UniPC4 CFG3 | Unsupported | 4417.63 · compiled, UniPC4 CFG3 | **4772.38** · BF16 NUMERIC, UniPC4 CFG3 |
| pi05 | 408.58 · NFE10 | 93.92 · compiled, NFE1 | Unsupported | **51.85** · FP8, NFE10 |
| GR00T N1.7 | 139.50 · NFE4 | 247.01 · native, NFE4 | Not qualified | **117.30** · native, NFE4 |
| DreamZero DROID | 23563.08 · fixed 8/16 DiT | Unsupported | 8729.64 · compiled, upstream step cache | **11899.42** · FP8, 16 solver updates, dynamic cache |

Precision, steps and cache policies are shown per cell; framework protocols differ.
VA measures early continuations. DreamZero's native reference uses an eager DiT with vendor encoder compilation.

—: unmeasured · Unsupported: no matching policy in the pinned registry · Not qualified: no validated measurement.
These are speed measurements; task quality is evaluated separately.

[Current protocol, raw results and historical comparison scope](eval/public_release_2026-09-15/results.rst) · [Reproduction commands](REPRODUCE.rst)

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
