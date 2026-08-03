<div align="center">

<img src="assets/instinctwm_2.png" alt="InstinctWM" width="360"/>

# InstinctWM

### Load, optimize, and deploy world-action models

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

[Optimization Stack](#optimization-stack) •
[Quick Start](#quick-start) •
[Evaluation](#evaluation) •
[Results](eval/lingbot_va_robotwin/RESULTS.md)

</div>

---

InstinctWM is an optimization framework for **world-action models** — robot policies that predict
what happens next *and* what to do about it in one model. It covers the full stack, from model
optimization down to hardware deployment. You describe the model once in a Backend Adapter;
InstinctWM determines which optimizations are legal, applies them, and reports what each one cost.

> **Status: early.** The evaluation pipeline, the measurement tooling, and the graph and cache
> layers are real and reproducible. The kernel and hardware layers are designed and being built.
> Every number here is measured on our own hardware with the scripts in [`eval/`](eval/).

---

## What's New 🔥

- **[2026/08]** 2/2-step inference passes paired non-inferiority on the *shipped* LingBot-VA
  checkpoint — **6.37× faster with no retraining**. 100 paired episodes, 10 tasks, pinned seeds.
- **[2026/08]** **3.38× bit-exact** on LingBot-VA in episode mode: 9585 → 2832 ms per control
  cycle, at `max |Δ action| = 0`. [Protocol and full chain →](eval/lingbot_va_robotwin/RESULTS.md)
- **[2026/08]** Remaining cost profiled. LingBot-VA *was* launch-bound; after graph capture it is
  GPU-bound again, which re-ranks every layer below.
- **[2026/08]** Canonical RoboTwin 2.0 baseline: **91.6% macro**, 50 tasks, 2500 episodes.
- **[2026/07]** Evaluation pipeline for LingBot-VA on RoboTwin 2.0, including a prompt-parity gate
  that closes a silent train/serve mismatch.

---

## Optimization Stack

InstinctWM organizes optimizations into six complementary layers, ordered by *what they change* —
from the model itself to the hardware it runs on.

| **MODEL** | **GRAPH** | **CACHE** | **ATTENTION** | **KERNEL** | **HARDWARE** |
|:--|:--|:--|:--|:--|:--|
| Step Reduction | **Prefill Extraction** | **KV Reuse** | FlashAttention | **Operator Fusion** | TensorRT |
| Parallel Decoding Distillation | **Execution Graph Rewrite** | **Cross-Attention Cache** | FlashInfer | *Fused AdaLN* | FP8 |
| rCM | **CUDA Graph Capture** | **Episode Cache** | Sana-Video Hybrid | *Triton Kernels* | INT8 |
| sCM | **Persistent State Analysis** | TeaCache | LongSana | Fused CFG | INT4 |
| DMD2 | **Static Memory Planning** | XCache | Linear Attention | Fused Scheduler | Jetson |
| DreamZero-Flash | Stream Overlap | SeaCache | Mamba / DeltaNet | Fused VAE | Thor |
| Latent Compression | ~~CFG Parallelization~~ | Window Cache | | Paged KV Kernels | Snapdragon |
| DC-AE / DC-VE | ~~Whole-Cycle Capture~~ | Energy-based Cache | | | |

**bold** shipped, gated bit-exact, measured end to end · *italic* implemented but not on the shipped
path · ~~struck~~ rejected *by measurement*, kept so it is not re-proposed · plain designed only.

**All 3.38× of measured speedup comes from GRAPH and CACHE.** The other four layers are either
unbuilt or, in the case of attention, deprioritized *by profile* — it is 7% of GPU busy, the item
intuition picks first and the measurement ranks near-last.

### What is measured today

| item | layer | status | evidence |
|---|---|---|---|
| Prefill extraction | graph | shipped | caches episode-constant cross-attention K/V for 30 layers; removes 89 of 226 TFLOP/cycle |
| Execution graph rewrite | graph | shipped | adapters publish sites, passes decide: `HoistInvariant`, `PromoteSmallOperand`, `ExplicitStepIndex` |
| CUDA Graph capture | graph | shipped | 1.21× whole-episode. The graph key does not converge — ~6 captures/cycle indefinitely |
| Persistent state analysis | graph | shipped | traces external reads/writes and graph-key fields; found two dependency bugs inspection had missed |
| Static memory planning | graph | shipped | reset clears logical state in place, behind a pointer certificate that fails closed |
| KV reuse | cache | shipped | ring addressing replaces a per-layer `mask.nonzero()` gather with an interval slice — the largest single step in the chain, and what makes capture legal at all |
| Episode cache | cache | shipped | reset isolation verified against a fresh server |
| Operator fusion framework | kernel | shipped | fusible regions, tier derivation, PTX-level assertions. It rejected three of our own kernels |
| Fused AdaLN | kernel | partial | removes the 35.4 MB activation upcast per block, bit-exact; the full norm+modulation fusion is not done |
| Triton kernels | kernel | partial | a bit-exact gated-residual kernel reaches 1.21–1.26× in microbenchmark, but is launch-dominated |
| CFG parallelization | graph | ruled out | the action stream's CFG branch is live on *both* liveness axes: output discarded, computation load-bearing |
| Whole-cycle capture | graph | ruled out | the KV read extent grows 152 slots/cycle, so the graph key cannot converge without changing numerics |

Measurements are LingBot-VA, episode mode: 45 consecutive control cycles, one reset.

---

## Quick Start

```bash
git clone https://github.com/general-instinct/InstinctWM && cd InstinctWM
pip install -e .                # analysis only: no torch, no GPU required
pip install -e ".[runtime]"     # to actually apply a plan and serve
```

Deciding *which* optimizations are legal is dependency-free by design, so you can inspect a plan on
a laptop. Only applying one needs torch.

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

---

## Supported Models

| model | status |
|:--|:--|
| **LingBot-VA** | Full runtime support. Primary optimization target and evaluation benchmark: 3.38× bit-exact in episode mode, with multi-episode bit-exactness, reset isolation, and pointer-stability gates. |
| **Cosmos3-Edge** | Engine support. One Plan runs under both executors, graph replay bit-exact against the eager oracle. **Plumbing only** — a torch-SDPA shim stands in for the served attention kernel, so no accuracy or speedup claim is made. |

Additional world-action models will be added over time. We would rather have two models fully
verified than six partly claimed.

---

## Documentation

| | |
|:--|:--|
| [Evaluation harness](eval/lingbot_va_robotwin/README.md) | Running the RoboTwin pipeline, and seven ways it can silently produce a plausible wrong number |
| [Results](eval/lingbot_va_robotwin/RESULTS.md) | Full optimization chain, per-pass measurements, measurement protocols |
| [`instinctwm/passes/`](instinctwm/passes/) | Pass interface: adapters publish sites, passes decide rewrites |
| [`instinctwm/engine/`](instinctwm/engine/) | Plan, executors, graph capture, dependency tracing |
| [`instinctwm/kernels/`](instinctwm/kernels/) | Fusion framework and tier derivation |

Per-layer documentation is being written and is not published yet.

---

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

---

## Examples

| | |
|:--|:--|
| [`probe_latency.py`](eval/lingbot_va_robotwin/probe_latency.py) | Short-horizon latency A/B across cumulative pass configurations |
| [`probe_episode.py`](eval/lingbot_va_robotwin/probe_episode.py) | Episode mode — consecutive cycles, one reset. The reporting standard |
| [`probe_bitexact.py`](eval/lingbot_va_robotwin/probe_bitexact.py) | Paired seeded rollouts at zero action delta |
| [`probe_cfg_liveness.py`](eval/lingbot_va_robotwin/probe_cfg_liveness.py) | Two-axis liveness test that ruled out CFG elision |
| [`serve_variant.py`](eval/lingbot_va_robotwin/serve_variant.py) | A/B policy server; every variant applies the same installers production does |
| [`certify_run.py`](eval/lingbot_va_robotwin/certify_run.py) | Paired non-inferiority certificate from per-episode JSONL |

---

## Roadmap

The chain is 9585 → 2832 ms, every step bit-exact, and all of it came from the graph and cache
layers. Those are now largely exhausted at this architecture: whole-cycle capture is structurally
blocked, CFG elision is illegal here, copy elimination has a 1.07× ceiling, and attention is 7% of
GPU busy.

GEMM time is now the dominant term, and **nothing bit-exact below the model layer touches it.** That
points at step reduction, which is why the next work is there rather than another runtime pass. The
2/2-step result suggests the capability boundary sits far from where the model runs today, so mapping
that boundary comes before optimizing it.

Current focus: model-layer step reduction, then the cache, kernel, and hardware layers.

---

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
