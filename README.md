<div align="center">

<img src="assets/iFlash.png" alt="InstinctFlash" width="360"/>

**A high-performance serving framework for robotics models.**

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

---

## What's new 🔥

- **[2026/09/17] RTX 5090 support.** Deploy on your workstation with the same Runtime API used on Jetson Thor. [Setup](INSTALL.rst#rtx-5090) · [Reproduce](REPRODUCE.rst#rtx-5090).
- **[2026/09/16] RTX 4090 support.** Desktop inference and WebSocket serving with dedicated installation profiles. [Setup](INSTALL.rst#rtx-4090) · [Reproduce](REPRODUCE.rst#rtx-4090).
- **[2026/09/15] Full-source release.** Eight robotics model families, acceleration kernels, and Python / WebSocket serving through one Runtime. [Get started](#install).
- **[2026/09/15] Jetson Thor benchmarks.** Up to **33.78×** speedup with LingBot-VA @2V/4A, using FP8 and fewer sampling steps. [Results](#results) · [Reproduce](REPRODUCE.rst).

## Results

Prediction p50 on **Jetson Thor** (ms), measured September 15, 2026.

We’ve seen up to **33.78× speedup** with no observed loss in task performance in our real-robot tests.

| Model | Acceleration line | PyTorch | InstinctFlash | Speedup |
|:--|:--|--:|--:|--:|
| LingBot-VA | FP8 · 25V/50A | 15506.32 | **2891.74** | **5.36×** |
| ↳ LingBot-VA | FP8 · 2V/4A | 2071.29 | **459.10** | **4.51×** |
| LingBot-VLA-4B | FP8 | 624.22 | **221.53** | **2.82×** |
| LingBot-VLA-V2-6B | FP8 | 734.56 | **394.11** | **1.86×** |
| Cosmos3 Edge DROID | NUMERIC · UniPC4 / CFG3 | 3393.78 | **1048.01** | **3.24×** |
| Cosmos3 Nano DROID | NUMERIC · UniPC4 / CFG3 | 10184.68 | **4772.38** | **2.13×** |
| pi05 | FP8 | 408.58 | **51.85** | **7.88×** |
| GR00T N1.7 | BITEXACT | 139.50 | **117.30** | **1.19×** |
| DreamZero DROID | FP8 · 16 steps · dynamic cache | 23563.08 | **11899.42** | **1.98×** |

VA measures early continuations; each row compares the same schedule.
The 33.78× headline includes 25V/50A → 2V/4A.
FP8 and sampling changes are optional.

[Protocol and raw results](eval/public_release_2026-09-15/results.rst) · [Native VA 2V/4A](eval/va_native_2v4a_2026-09-15/README.rst) · [Reproduction commands](REPRODUCE.rst)

## Install

```bash
git clone https://github.com/General-Instinct/InstinctFlash && cd InstinctFlash
python3 -m venv .venv-core
source .venv-core/bin/activate
python -m pip install . uv==0.12.5
```

The Python 3.10+ core inspects checkpoints and plans without PyTorch or a GPU.
Inference uses a separate, pinned environment for each model family. For RTX 4090:

```bash
python3 scripts/bootstrap_vendor.py install pi05 --target rtx4090 \
  --python python3.12 --root ~/ifl-pi05-4090 --ptxas /usr/local/cuda/bin/ptxas
source ~/ifl-pi05-4090/activate.sh
```

Use `va`, `vla4`, `vla2`, `pi05`, `groot`, `edge`, `nano` or `dreamzero`.
Edge and Nano use Python 3.13; the other families use Python 3.12.
The bootstrap installs the upstream source, compatibility patches, core and adapter.
Model weights are downloaded separately. See [RTX 5090 setup](INSTALL.rst#rtx-5090),
[RTX 4090 setup](INSTALL.rst#rtx-4090)
or [Jetson Thor setup](INSTALL.rst#jetson-thor), which selects `--target jetson_thor`
and uses the [Thor CUDA backend build](serving/README.rst).

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

`observation` is a dict in the model's own format; `result["action"]` contains its action array.
For LingBot-VA, pass `executed_action=...` when the controller changes a predicted action
chunk, so the next prediction uses the actions actually executed.

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

After the [vendor and auxiliary-asset preparation](REPRODUCE.rst), reproduce paired
eager/default/selected Runtime measurements with the included inputs and fixed
checkpoint revision. Thor also requires its [native backend](serving/README.rst).
Keep the model and asset environments activated. For RTX 4090:

```bash
python -I -m benchmarks.regression.reproduce prepare --target rtx4090 \
  --model pi05 --mode fp8 --output pi05-inputs
python -I -m benchmarks.regression.reproduce run --prepared pi05-inputs --output pi05-results
python -I -m benchmarks.regression.serve_smoke --prepared pi05-inputs --output pi05-serving
```

`run` writes checked JSON/CSV reports and full action arrays. `serve_smoke` tests
the actual CLI and WebSocket pipeline across two episodes. Use `--mode native`
for default precision; FP8, numerical compilation and changed schedules are
explicit selections. [Reproduction guide](REPRODUCE.rst).
For additional framework comparisons, use the [pinned comparison recipes](benchmarks/regression/FRAMEWORK_COMPARISON.rst).

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

## Acknowledgements

We thank the following projects and their contributors for the code, models, tools,
and ideas that InstinctFlash builds on:

- **[Cosmos](https://github.com/NVIDIA/cosmos-framework)**,
  **[DreamZero](https://github.com/dreamzero0/dreamzero)**,
  **[Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)**,
  **[LingBot-VA](https://github.com/robbyant/lingbot-va)**,
  **[LingBot-VLA](https://github.com/robbyant/lingbot-vla)**, and
  **[LingBot-VLA-V2](https://github.com/robbyant/lingbot-vla-v2)** — upstream model
  implementations and checkpoints.
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** and
  **[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)** — attention kernels,
  matrix-multiplication templates, and supporting backend code.
- **[FlashRT](https://github.com/flashrt-project/FlashRT)** — substantial portions
  of our `serving/` backend are derived from and adapted from FlashRT, including
  runtime components, CUDA kernels, and weight-processing utilities.
- **[Hugging Face](https://huggingface.co)**,
  **[PyTorch](https://github.com/pytorch/pytorch)**, and
  **[Triton](https://github.com/triton-lang/triton)** — framework, compiler, and model tooling.
- **[LeRobot](https://github.com/huggingface/lerobot)** — model implementations,
  preprocessing, and robotics tooling.
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)**,
  **[RoboLab](https://github.com/NVlabs/RoboLab)**, and
  **[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)** — simulation environments and
  evaluation infrastructure.
- **[msgpack-numpy](https://github.com/lebedov/msgpack-numpy)** and
  **[OpenPI](https://github.com/Physical-Intelligence/openpi)** — π0/π0.5 model code
  and client serialization from OpenPI, whose serialization implementation adapts
  msgpack-numpy.
- **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** — design references for
  KV-cache and conditioning-cache management.

Third-party code and model assets remain subject to their respective licenses.
See the license and attribution notices accompanying each component.
