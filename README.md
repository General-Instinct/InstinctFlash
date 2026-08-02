<div align="center" id="instinctwm-top">
<img src="assets/instinctwm_2.png" alt="InstinctWM Logo" width="400" margin="10px"></img>
<h3 align="center">Load, optimize, and deploy world-action models</h3>

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

<p align="center">
| <a href="#the-optimization-stack"><b>Optimization Stack</b></a> | <a href="eval/lingbot_va_robotwin/README.md"><b>Evaluation</b></a> | <a href="eval/lingbot_va_robotwin/RESULTS.md"><b>Results</b></a> |
</p>

---

*Latest News* 🔥

- [2026/08] **3.38x bit-exact on LingBot-VA**, episode mode: 9585 to 2832 ms per control cycle over 45 consecutive cycles with a single reset, verified at `max |delta action| = 0` on paired seeded rollouts.
  A short reset-based probe reports 6.96x on the same build. That protocol resets between repeats, which rewinds the KV ring so each repeat replays graphs the discarded first run captured — it converts a per-cycle cost into a warm-up cost and **overstates this chain by 2.13x**. Episode mode is the reporting standard; the short-horizon number is kept only for continuity with earlier posts.
- [2026/08] **Profiled the remaining cost.** LingBot-VA *was* launch-bound: an aten op cost 6.2 us whether it touched 1 element or 1 MB, and 83.6% of that was `cudaLaunchKernel` itself. Graph capture moved the launches inside graphs, and the workload is now GPU-bound again — so the profile that motivated the launch work no longer describes the current default. Launch counts quoted from that era are pre-capture and should not be read as current.
- [2026/08] **Canonical RoboTwin 2.0 baseline: 91.6% macro** across all 50 tasks and 2500 episodes, zero failures.
- [2026/08] Optimizer skeleton landed. Passes fire from adapter declarations, not flags, and carry equivalence tiers that do not compose upward.
- [2026/07] Evaluation pipeline for LingBot-VA on RoboTwin 2.0, including a prompt-parity gate that closes a silent train/serve mismatch nobody upstream was checking.

---

## About

**InstinctWM** is an optimization framework for **world-action models**: the class of robot
policies that predict what happens next *and* what to do about it, in one model.

You write a Backend Adapter that states facts about your model. InstinctWM works out which
optimizations are legal, applies them, and tells you what each one cost in accuracy.

```python
from instinctwm import load, Optimizer, Tier

model = load("lingbot-va-posttrain-robotwin")          # the adapter states facts
plan  = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
print(plan.explain())                                  # what fired, and why
server = plan.serve(model, port=29056)                 # deploy
```

> Early. The eval pipeline, the measurement tooling, and the first optimizer passes are real and
> reproducible. The kernel and compiler layers are designed and being built. Every number quoted
> here is measured on our hardware and reproducible with the scripts in `eval/`.

### How it works

Bottom-up. A layer owns something only if **at least two world-action models need it identically**.

| layer | owns |
|---|---|
| Model-level | adaptive NFE against a deadline, step and velocity caching, world-token budget, async closed-loop execution |
| Graph and compiler | loop-invariant hoisting, dead-output elimination, guidance elision, shape specialization, graph capture |
| Cache and state | paged KV arena with lifetimes, commit transactions, multi-tower caches, session park and unpark |
| Attention | fused write-then-attend over a block table; windows, sinks, cross-attention, guidance branches |
| Triton and CUDA | fused norm, modulation, QKV and RoPE; fused FFN; pattern-matched onto the module tree |
| Hardware | capability probe and dispatch across H100/H200, Ada, and Blackwell (5090, Jetson Thor) |

Every optimization is classified by how it is discovered:

- **AUTO**, detected with no help from the module tree, a trace, a profile, or a differential test.
  This is the product.
- **DECLARED**, needs a fact that cannot be safely inferred. A dtype keep-list is one: guess wrong
  and you get silently wrong actions.
- **CHECKPOINT**, needs new weights, and is therefore something we can host but not deliver.

### What is actually supported

Two models, both validated end to end on this hardware:

| model | status |
|---|---|
| **LingBot-VA** | primary optimization target and evaluation benchmark. 3.38x bit-exact, episode mode. Full correctness gates: multi-episode bit-exactness, reset isolation, pointer stability. |
| **Cosmos3-Edge** | second reference model, used to validate that the engine generalizes. One Plan runs under both executors with graph replay bit-exact against the eager oracle. **Plumbing only** — a torch-SDPA shim stands in for the served attention kernel, so no accuracy or speedup claim is made. |

Everything else is **future work**. The state descriptors carry unvalidated design entries for
other model families; those are design sketches, not support, and nothing has been measured on
them. We would rather have two models fully verified than six partly claimed.

One design finding does generalize and is worth keeping: **"stateless VLA versus stateful WAM" is
a false dichotomy** — KV persistence is a lifetime field (`none`, `chunk`, `window`, `episode`),
not a boolean. Cosmos3-Edge is the validated instance of the far end of that axis: it keeps no KV
pool at all, and the same runtime serves it with no `if is_vla` anywhere.

### Getting started

```bash
git clone https://github.com/general-instinct/InstinctWM && cd InstinctWM
cd eval/lingbot_va_robotwin && source ./env.sh

IWM_FA_SHIM=1 ./servers.sh start 8        # one policy server per GPU
$IWM_SERVER_PY check_prompt_parity.py ... # the correctness gate, run it first
./run_eval.sh myrun 50 adjust_bottle ...  # fan tasks across the fleet
$IWM_CLIENT_PY aggregate.py $IWM_RESULT_DIR/myrun --expect-episodes 50
```

`aggregate.py` prints `REPORTABLE: NO` and refuses to give you a number when the run is incomplete
or internally inconsistent. That is deliberate.

New contributors should start with
[eval/lingbot_va_robotwin/README.md](eval/lingbot_va_robotwin/README.md). It documents seven ways
this pipeline can silently produce a plausible wrong number, every one of which we hit. The most
instructive: LingBot-VA **never runs T5 during training**, it reads a precomputed embedding baked
into the dataset, while the server recomputes it live, and nothing upstream checked that the two
agree. An earlier project lost a 22.7-hour run to exactly that class of bug. `check_prompt_parity.py`
closes it, and passes bit-exactly.

The house rule follows: **a number you cannot defend is worse than no number.** Label plumbing as
plumbing. Assert every knob you set. Read the server log.

Good first work is in [The optimization stack](#the-optimization-stack) below. It is organized by
layer, and every item carries its status — shipped, partial, ruled out, or future — so it is clear
what is real and what is only designed.

## The optimization stack

InstinctWM is organized as six optimization layers, not as a list of tricks. A layer is defined by
*what it changes*: Layer 1 changes the model, Layer 6 changes the hardware target, and correctness
gets progressively harder to guarantee as you go up.

Status is per-item and means exactly this:

| | meaning |
|---|---|
| **shipped** | implemented, gated bit-exact, and measured end to end on LingBot-VA |
| **partial** | implemented and measured, but not on the shipped path — reason given |
| **ruled out** | tried or tested, and rejected *by measurement*. Kept so it is not re-proposed |
| **future** | designed or surveyed only. Nothing implemented, nothing measured |

All measurements are LingBot-VA, episode mode (45 consecutive control cycles, one reset) unless
noted. Cosmos3-Edge results are plumbing-only and never carry a speedup claim.

---

### Layer 1 — Model-level

Changes what the model computes. The largest lever available and the only one that touches the
NFE count, which the profile says is the dominant term. Nothing here is implemented.

**Step reduction** — `future`
Parallel Decoding Distillation · rCM · sCM · DMD2 · DreamZero-Flash

**Latent compression** — `future`
DC-AE / DC-VE · other latent tokenizer variants

> Every item requires new weights, so none is behavior-preserving. Layer 1 is where the remaining
> order-of-magnitude is, and also where the equivalence tier drops to `BEHAVIORAL` — it needs the
> accuracy harness, not the bit-exact gate.

---

### Layer 2 — Graph-level

Changes *when and how* work is issued, never what is computed. Everything shipped so far lives
here, and it is the layer where bit-exactness is achievable.

| item | status | evidence |
|---|---|---|
| Prefill extraction | **shipped** | P002: caches episode-constant cross-attention K/V for 30 layers, removes 89 of 226 TFLOP/cycle |
| Execution graph rewrite | **shipped** | pass framework: `HoistInvariant`, `PromoteSmallOperand`, `ExplicitStepIndex` — adapters publish sites, passes decide |
| Persistent state analysis | **shipped** | `engine/deps.py` derives external reads/writes, host mutations and graph-key fields by tracing. Found two dependency bugs inspection had missed twice |
| Static memory planning | **shipped** | P006: reset clears logical state in place; pointer certificate fails closed |
| CUDA Graph capture | **shipped** | P005: 1.21x whole-episode. **Caveat:** the key does not converge — ~6 captures/cycle indefinitely |
| Prefill cache | **shipped** | same mechanism as prefill extraction |
| CFG parallelization | **ruled out** | a two-axis liveness test found the action stream's CFG branch 1 live on *both* axes (output 5.64, shared-state 5.39, vs 1.03 movement). Output discarded, computation load-bearing |
| Whole-cycle capture | **ruled out** | blocked structurally: the KV read extent grows 152 slots/cycle, so the graph key cannot converge without changing numerics |
| Stream overlap | **future** | attacks the FIXED cost term; matters most at low NFE |

---

### Layer 3 — Cache

Reuses computation across steps or episodes. Partly shipped.

| item | status | evidence |
|---|---|---|
| KV reuse | **shipped** | P003 ring KV: replaces a per-layer `mask.nonzero()` gather with an interval slice. Largest single step in the chain, and what makes graph capture legal at all |
| Cross-attention cache | **shipped** | P002 |
| Episode-level cache | **shipped** | P006, with reset isolation verified against a fresh server |
| TeaCache · XCache · SeaCache · energy-based cache | **future** | step-skipping caches; all trade behavior for speed and need the accuracy harness |
| Window cache | **future** | |

---

### Layer 4 — Attention

**Deprioritized by measurement, not by preference.** Attention is **7% of GPU busy** on the current
default. It is the item intuition picks first and the profile ranks near-last.

Sana-Video hybrid attention · LongSana · linear attention · Mamba / DeltaNet · FlashAttention ·
FlashInfer kernels — all `future`.

> FlashInfer's Init–Plan–Run split already shaped the engine's design (plan on the host into GPU
> buffers, keep the run phase shape-static and capture-safe) even though no FlashInfer kernel is
> integrated.

---

### Layer 5 — Kernel

| item | status | evidence |
|---|---|---|
| Operator fusion framework | **shipped** | `kernels/`: fusible regions, tier derivation, PTX-level assertions. It rejected three of our own kernels |
| Fused AdaLN (modulation) | **partial** | `PromoteSmallOperand` removes the 35.4 MB activation upcast per block, bit-exact. The full norm+modulation fusion is not done |
| Triton kernels | **partial** | a bit-exact gated-residual kernel exists at 1.21–1.26x in a microbenchmark and is **not shipped**: it is launch-dominated, and Triton's Python launcher costs 11.0 us against PyTorch's 6.2 us dispatch |
| Fused CFG · fused scheduler · fused VAE · paged-KV kernels | **future** | |

> The measured lesson: after graph capture, a fused kernel competes against ~1.17 us of GPU-side
> launch latency, not 6.2 us of dispatch. Fusion should be counted in *kernels removed inside the
> graph*, and the copy audit puts the whole remaining copy traffic at a 1.07x ceiling.

---

### Layer 6 — Hardware

Nothing implemented. All `future`.

TensorRT · CUDA Graph backends beyond ours · FP8 · INT8 · INT4 · Jetson · Thor · Snapdragon

> Quantization is Layer 6 and deliberately unprioritized: it attacks bytes and FLOPs, which the
> profile puts at roughly 23% of the problem. It is not free either — everything here is
> `NUMERIC` at best.

---

### What the stack says about what to do next

The chain is **9585 → 2832 ms, 3.38x, every step bit-exact**, and all of it came from Layer 2 and
Layer 3. Those layers are now largely exhausted at the current architecture: whole-cycle capture is
structurally blocked, CFG elision is illegal here, copy elimination has a 1.07x ceiling, and
attention is 7% of GPU busy.

GEMM time is now the dominant term, and **nothing bit-exact on Layers 2–5 touches it.** That points
at Layer 1 — fewer steps — which is why the next work is a Layer 1 design study rather than another
runtime pass.

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

AGPLv3 is a network-copyleft licence: if you run a modified InstinctWM as a service, you must
offer the modified source to its users. Third-party components we build on keep their own terms
(vLLM-Omni is Apache-2.0, which is compatible in this direction).
