<div align="center" id="instinctwm-top">
<img src="assets/instinctwm_2.png" alt="InstinctWM" width="400"></img>
<h3 align="center">Load, optimize, and deploy world-action models</h3>

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

<p align="center">
| <a href="#optimization-stack"><b>Optimization Stack</b></a> | <a href="#quick-start"><b>Quick Start</b></a> | <a href="#evaluation"><b>Evaluation</b></a> | <a href="eval/lingbot_va_robotwin/RESULTS.md"><b>Results</b></a> |
</p>

---

InstinctWM is an inference and optimization framework for **world-action models** — robot policies
that predict what happens next *and* what to do about it in one model. You describe the model once
in a Backend Adapter; InstinctWM determines which optimizations are legal, applies them, and reports
what each one cost in accuracy.

> **Status: early.** The evaluation pipeline, the measurement tooling, and the Layer 2–3 optimizer
> passes are real and reproducible. The kernel and hardware layers are designed and being built.
> Every number here is measured on our own hardware with the scripts in [`eval/`](eval/).

## What's New

- **[2026/08]** 2/2-step inference passes paired non-inferiority on the *shipped* LingBot-VA
  checkpoint — **6.37× faster with no retraining**. 100 paired episodes, 10 tasks, pinned seeds.
- **[2026/08]** **3.38× bit-exact** on LingBot-VA in episode mode: 9585 → 2832 ms per control
  cycle, at `max |Δ action| = 0`. [Protocol and full chain →](eval/lingbot_va_robotwin/RESULTS.md)
- **[2026/08]** Remaining cost profiled. LingBot-VA *was* launch-bound; after graph capture it is
  GPU-bound again, which re-ranks every layer below.
- **[2026/08]** Canonical RoboTwin 2.0 baseline: **91.6% macro**, 50 tasks, 2500 episodes.
- **[2026/07]** Evaluation pipeline for LingBot-VA on RoboTwin 2.0, including a prompt-parity gate
  that closes a silent train/serve mismatch.

## Optimization Stack

InstinctWM is organized as six optimization layers rather than a list of tricks. A layer is defined
by *what it changes* — and correctness gets harder to guarantee the further left you go.

```
     changes what is computed ────────────────────────── changes where it runs

     ┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
     │   MODEL   │   GRAPH   │   CACHE   │ ATTENTION │  KERNEL   │ HARDWARE  │
     │    L1     │    L2     │    L3     │    L4     │    L5     │    L6     │
     ├───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
     │ what the  │ when work │ what gets │ how tokens│ how one   │ what it   │
     │ model     │ is issued │ recomputed│ mix       │ kernel is │ executes  │
     │ computes  │           │           │           │ written   │ on        │
     ├───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
     │BEHAVIORAL │ BITEXACT  │ BITEXACT  │  NUMERIC  │ BITEXACT  │  NUMERIC  │
     ├───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
     │  future   │  shipped  │  shipped  │  future   │  partial  │  future   │
     └───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
                 └─── 3.38× bit-exact ───┘
```

Every shipped speedup so far comes from Layers 2 and 3. Status is per item — **shipped** (gated
bit-exact and measured end to end) · **partial** (measured, but not on the shipped path) · **ruled
out** (rejected *by measurement*, kept so it is not re-proposed) · **future** (designed only,
nothing measured). Measurements are LingBot-VA, episode mode: 45 consecutive control cycles, one
reset.

### L1 · Model

Changes what the model computes. The only layer that touches the NFE count — the dominant term in
the current profile — and the only one that needs new weights, so nothing here is behavior
preserving. Gated by the accuracy certificate, not the bit-exact gate.

| item | status |
|---|---|
| Step reduction — PDD · rCM · sCM · DMD2 · DreamZero-Flash | **future** |
| Latent compression — DC-AE / DC-VE | **future** |

### L2 · Graph

Changes *when and how* work is issued, never what is computed. Where bit-exactness is achievable.

| item | status | evidence |
|---|---|---|
| Prefill extraction | **shipped** | caches episode-constant cross-attention K/V for 30 layers; removes 89 of 226 TFLOP/cycle |
| Execution graph rewrite | **shipped** | adapters publish sites, passes decide: `HoistInvariant`, `PromoteSmallOperand`, `ExplicitStepIndex` |
| Persistent state analysis | **shipped** | traces external reads/writes and graph-key fields; found two dependency bugs inspection had missed |
| Static memory planning | **shipped** | reset clears logical state in place, behind a pointer certificate that fails closed |
| CUDA Graph capture | **shipped** | 1.21× whole-episode. Caveat: the graph key does not converge — ~6 captures/cycle indefinitely |
| CFG parallelization | **ruled out** | the action stream's CFG branch is live on *both* liveness axes: output discarded, computation load-bearing |
| Whole-cycle capture | **ruled out** | structurally blocked — the KV read extent grows 152 slots/cycle, so the key cannot converge without changing numerics |
| Stream overlap | **future** | attacks the FIXED cost term; matters most at low NFE |

### L3 · Cache

Reuses computation across steps or episodes.

| item | status | evidence |
|---|---|---|
| KV reuse | **shipped** | ring addressing replaces a per-layer `mask.nonzero()` gather with an interval slice — the largest single step in the chain, and what makes capture legal at all |
| Cross-attention cache | **shipped** | same mechanism as prefill extraction |
| Episode-level cache | **shipped** | reset isolation verified against a fresh server |
| Step-skipping caches — TeaCache · XCache · SeaCache | **future** | trade behavior for speed; need the accuracy certificate |

### L4 · Attention

**Deprioritized by measurement, not preference.** Attention is **7% of GPU busy** on the current
default — the item intuition picks first, and the profile ranks near-last.

All `future`: Sana-Video hybrid · LongSana · linear attention · Mamba / DeltaNet · FlashAttention ·
FlashInfer. That said, FlashInfer's Init–Plan–Run split already shaped the engine's design — plan on
the host into GPU buffers, keep the run phase shape-static and capture-safe.

### L5 · Kernel

| item | status | evidence |
|---|---|---|
| Fusion framework | **shipped** | fusible regions, tier derivation, PTX-level assertions. It rejected three of our own kernels |
| Fused AdaLN | **partial** | removes the 35.4 MB activation upcast per block, bit-exact; the full norm+modulation fusion is not done |
| Triton kernels | **partial** | a bit-exact gated-residual kernel reaches 1.21–1.26× in microbenchmark and is **not shipped** — it is launch-dominated |
| Fused CFG · scheduler · VAE · paged-KV | **future** | |

After graph capture a fused kernel competes against ~1.17 µs of GPU-side launch latency, not 6.2 µs
of host dispatch. Fusion has to be counted in *kernels removed inside the graph*.

### L6 · Hardware

All `future`: TensorRT · FP8 · INT8 · INT4 · Jetson · Thor · Snapdragon. Quantization sits here and
is deliberately unprioritized — it attacks bytes and FLOPs, roughly 23% of the current problem, and
is `NUMERIC` at best.

## Quick Start

```bash
git clone https://github.com/general-instinct/InstinctWM && cd InstinctWM
pip install -e .                # analysis only: no torch, no GPU required
pip install -e ".[runtime]"     # to actually apply a plan and serve
```

Deciding *which* optimizations are legal is dependency-free by design, so you can inspect a plan on
a laptop. Only applying one needs torch. The adapter states facts about the model; the optimizer
decides what follows from them.

```python
from instinctwm import load, Optimizer, Tier

model  = load("lingbot-va-posttrain-robotwin")
plan   = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
print(plan.explain())                    # what fired, and why
server = plan.serve(model, port=29056)   # deploy
```

`plan.explain()` reports every decision, including the ones it declined:

```
InstinctWM plan for lingbot-va-posttrain-robotwin      plan tier: BITEXACT

  APPLY  fsdp_elision            [BITEXACT] world_size=1, so every FSDP all-gather
                                            is identity while still paying a
                                            flat-param copy and stream sync
  APPLY  allocator_churn_elision [BITEXACT] closed-loop serving has a stable
                                            working set
  ...
```

## Supported Models

Two models, both validated end to end on our hardware.

| model | status |
|---|---|
| **LingBot-VA** | Primary optimization target and evaluation benchmark. 3.38× bit-exact in episode mode, with multi-episode bit-exactness, reset isolation, and pointer-stability gates. |
| **Cosmos3-Edge** | Second reference model, used to verify the engine generalizes. One Plan runs under both executors, graph replay bit-exact against the eager oracle. **Plumbing only** — a torch-SDPA shim stands in for the served attention kernel, so no accuracy or speedup claim is made. |

Everything else is future work. We would rather have two models fully verified than six partly
claimed. Cosmos3-Edge keeps no KV pool at all and the same runtime serves it with no `if is_vla`
anywhere, which is the property that makes the engine model-agnostic.

## Documentation

| | |
|---|---|
| [Evaluation harness](eval/lingbot_va_robotwin/README.md) | How to run the RoboTwin pipeline, and seven ways it can silently produce a plausible wrong number |
| [Results](eval/lingbot_va_robotwin/RESULTS.md) | Full optimization chain, per-pass measurements, and measurement protocols |
| [`instinctwm/passes/`](instinctwm/passes/) | Pass interface: adapters publish sites, passes decide rewrites |
| [`instinctwm/engine/`](instinctwm/engine/) | Plan, executors, graph capture, dependency tracing |
| [`instinctwm/kernels/`](instinctwm/kernels/) | Fusion framework and tier derivation |

## Evaluation

Accuracy is gated separately from speed, and neither gate trusts the other.

- **Bit-exact optimizations** are gated at `max |Δ action| = 0` on paired seeded rollouts.
- **Behavior-changing optimizations** are gated by paired non-inferiority on pinned seeds, with the
  margin declared *before* the run, exact McNemar on discordant pairs, and a per-task table.

```bash
cd eval/lingbot_va_robotwin && source ./env.sh

IWM_FA_SHIM=1 ./servers.sh start 8         # one policy server per GPU
$IWM_SERVER_PY check_prompt_parity.py ...  # correctness gate — run it first
./run_eval.sh myrun 50 adjust_bottle ...   # fan tasks across the fleet
$IWM_CLIENT_PY aggregate.py $IWM_RESULT_DIR/myrun --expect-episodes 50
```

`aggregate.py` prints `REPORTABLE: NO` and refuses to give a number when a run is incomplete or
internally inconsistent. That is deliberate: **a number you cannot defend is worse than no number.**

The gate that earns its keep most often is `check_prompt_parity.py`. LingBot-VA never runs T5 during
training — it reads a precomputed embedding baked into the dataset, while the server recomputes it
live, and nothing upstream checked that the two agree. An earlier project lost a 22.7-hour run to
exactly that class of bug.

## Examples

| | |
|---|---|
| [`probe_latency.py`](eval/lingbot_va_robotwin/probe_latency.py) | Short-horizon latency A/B across cumulative pass configurations |
| [`probe_episode.py`](eval/lingbot_va_robotwin/probe_episode.py) | Episode mode — consecutive cycles, one reset. The reporting standard |
| [`probe_bitexact.py`](eval/lingbot_va_robotwin/probe_bitexact.py) | Paired seeded rollouts at `max abs delta action = 0` |
| [`probe_cfg_liveness.py`](eval/lingbot_va_robotwin/probe_cfg_liveness.py) | Two-axis liveness test that ruled out CFG elision |
| [`serve_variant.py`](eval/lingbot_va_robotwin/serve_variant.py) | A/B policy server; every variant applies the same installers production does |
| [`certify_run.py`](eval/lingbot_va_robotwin/certify_run.py) | Paired non-inferiority certificate from per-episode JSONL |

## Roadmap

The chain is 9585 → 2832 ms, every step bit-exact, and all of it came from Layers 2 and 3. Those
layers are now largely exhausted at this architecture: whole-cycle capture is structurally blocked,
CFG elision is illegal here, copy elimination has a 1.07× ceiling, and attention is 7% of GPU busy.

GEMM time is now the dominant term, and **nothing bit-exact on Layers 2–5 touches it.** That points
at Layer 1 — fewer steps — which is why the next work is step reduction rather than another runtime
pass. The 2/2-step result suggests the capability boundary sits far from where the model runs today,
so mapping that boundary comes before optimizing it.

## Citation

```bibtex
@software{instinctwm2026,
  title  = {InstinctWM: Load, optimize, and deploy world-action models},
  author = {General Instinct},
  year   = {2026},
  url    = {https://github.com/general-instinct/InstinctWM}
}
```

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

AGPLv3 is a network-copyleft licence: if you run a modified InstinctWM as a service, you must offer
the modified source to its users. Third-party components keep their own terms (vLLM-Omni is
Apache-2.0, compatible in this direction).
