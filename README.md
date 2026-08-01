<div align="center" id="instinctwm-top">
<img src="assets/instinctwm_2.png" alt="InstinctWM Logo" width="400" margin="10px"></img>
<h3 align="center">Load, optimize, and deploy world-action models</h3>

[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Website](https://img.shields.io/badge/Website-general--instinct.com-000000.svg)](https://general-instinct.com/)
[![YC](https://img.shields.io/badge/Y%20Combinator-P26-orange.svg)](https://www.ycombinator.com/companies/general-instinct)

</div>

<p align="center">
| <a href="#roadmap"><b>Roadmap</b></a> | <a href="eval/lingbot_va_robotwin/README.md"><b>Evaluation</b></a> | <a href="eval/lingbot_va_robotwin/RESULTS.md"><b>Results</b></a> |
</p>

---

*Latest News* 🔥

- [2026/08] **First bit-exact speedup: 2.16x on LingBot-VA** (8881 to 4115 ms per control cycle, 3.6 to 7.8 Hz), verified at `max |delta action| = 0` on paired seeded rollouts.
- [2026/08] **Profiled the remaining cost.** LingBot-VA is launch- and gather-bound, not compute-bound: the GPU is idle 51% of the cycle, real arithmetic is 8.5% of wall clock, and one control step issues 469,811 kernel launches.
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
> reproducible. The kernel and compiler layers are designed and being built. Every number below is
> measured on our hardware and reproducible with the scripts in `eval/`.

### Why this exists

A world-action model is not an LLM and not an image generator, and the difference is the whole
problem.

**It runs a closed loop with a deadline.** A bimanual manipulator wants 16 to 30 Hz. Miss the
deadline and the robot does not get a slower answer, it gets a *wrong* one, because the world moved
while you were thinking.

**It is latency-bound at batch 1, not throughput-bound.** One robot, one episode, one observation
at a time. Most serving machinery exists to fill a GPU with concurrent requests. Robots do not have
that problem.

**And it is far slower than it needs to be.** We measured a state-of-the-art open WAM,
[LingBot-VA](https://github.com/robbyant/lingbot-va), on an H100: **3.6 Hz**, roughly 36x off its
own memory-bandwidth roofline. When we profiled it, the GPU was **idle 51% of the cycle**, real
arithmetic was **8.5% of wall clock**, and it issued **469,811 kernel launches per control step**.
No individual kernel was slow. Almost all of it was overhead.

That gap is not one model's bug. It is what happens when research code meets a control loop, and it
recurs with every new WAM because nobody re-does the work.

### What is fundamentally new

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) is genuinely good and we build on it. We
adopt their component-discovery contract, their AR-diffusion KV spec, their cross-attention cache
design, and we fork their paged attention kernel. This is not an attempt to be different from them.

Two things are missing from every WAM stack shipping today, theirs included.

**1. Optimizations are configured, not derived.** `num_inference_steps` is fixed at construction.
Step-cache schedules are hardcoded per deploy file. Every model re-implements its own denoise loop:
across vLLM-Omni's 47 model pipelines, 27 define their own `prepare_latents`, 31 call
`scheduler.step` in-pipeline, 33 implement `encode_prompt`. Because the loop belongs to the model,
the runtime cannot vary NFE mid-episode, preempt between forwards, or capture graphs uniformly.

InstinctWM inverts that. **The adapter declares facts; the optimizer derives optimizations.** An
author writes "the action stream uses positive-only guidance" and "the text conditioning is a pure
function of the instruction". They never write "skip the negative branch" or "cache the
cross-attention K/V". Those are *derived*, and they fire on any future model whose declarations
imply them.

**2. Nobody measures what a speedup costs.** The two largest published WAM speedups are both lossy:
DreamZero's DiT caching reports *no accuracy number at all* for its 16-to-4 step reduction, and
DreamZero-Flash's entire steps-versus-success evidence is three points on one task at n=10 with
overlapping error bars. vLLM-Omni's repository contains no success-rate gates.

InstinctWM ties every optimization to a **closed-loop robotics benchmark result**. Each pass carries
an equivalence tier, and **tiers do not compose upward**:

| tier | claim | cost to establish |
|---|---|---|
| `BITEXACT` | `torch.equal` on per-step latents and committed K/V, in the production kernel config | free, ships with no benchmark run |
| `NUMERIC` | bounded norm of the delta, justified by a *named* structural invariant | cheap |
| `BEHAVIORAL` | changes outputs | paired non-inferiority run against a measured noise floor, roughly 10x the GPU time |

One `BEHAVIORAL` pass makes the whole plan `BEHAVIORAL`. `plan.bitexact_subset()` returns the
largest configuration you can ship without buying a benchmark run.

### Results so far

8x H100, LingBot-VA on RoboTwin 2.0, reproducible from `eval/lingbot_va_robotwin/`.

Accuracy baseline, 50 tasks x 50 episodes, 2500 episodes, zero failures:

| | |
|---|---|
| macro (leaderboard definition) | **91.6%** |
| micro pooled | 91.6% [90.5, 92.7] |
| published reference | 92.9% easy / 91.6% hard |

Latency, batch 1, idle H100, one control cycle of 32 actions:

| configuration | cycle | rate | accuracy |
|---|---|---|---|
| stock | 8881 ms | 3.6 Hz | reference |
| + substrate passes | 4624 ms | 6.9 Hz | **bit-exact**, `max abs delta = 0` |
| + conditioning prefill | **4115 ms** | **7.8 Hz** | **bit-exact**, `max abs delta = 0` |

**2.16x, provably free.** Not "within noise": *zero* delta, against a reference whose own
chunk-to-chunk action movement is 1.03.

The most useful result is the one that underperformed. Conditioning prefill removes **39% of all
arithmetic** and bought **1.05x**. That sent us to the profiler, and it is why quantization sits low
on our roadmap while KV addressing sits at the top.

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
- **DECLARED**, needs a fact that cannot be safely inferred. Pi-0's fp32 keep-list is one: guess
  wrong and you get silently wrong actions.
- **CHECKPOINT**, needs new weights. DreamZero-Flash is a training recipe, `Beta(7,1)`
  video-timestep sampling, not a runtime trick. We can host it; we cannot deliver it.

The design is validated against six model families, chosen because they disagree with each other:
LingBot-VA, DreamZero, Cosmos3-Edge, InternVLA-A1, GR00T N, and pi-0/pi-0.5. One finding reshaped
the abstraction: **"stateless VLA versus stateful WAM" is a false dichotomy.** Pi-0 builds a prefix
KV cache, commits it, reads it from all 10 denoise forwards, and drops it, which is structurally
identical to LingBot-VA's episode-scoped stream. They differ only in **lifetime**. So KV persistence
is a lifetime field (`none`, `chunk`, `window`, `episode`), not a boolean, and one runtime serves
both with no `if is_vla` anywhere.

The cross-model derivation behind that abstraction, the full profile, and the prioritized
low-level work are summarised in [Roadmap](#roadmap) below.

### Honest status

What is real: the eval pipeline, the 91.6% baseline, the 2.16x bit-exact speedup, the profile, and
the optimizer skeleton with five working passes.

What is designed but not built: the paged KV arena, the fused kernels, graph capture, and the
deadline governor.

What we do not have: a training stack, so anything requiring new weights is out of reach. NVFP4 is
Blackwell-only, so the largest quantization win is unavailable on H100 and 4090. And our own layered
design has already been partly falsified by adversarial review against Cosmos3-Edge and GR00T. The
corrections are folded into the roadmap below. We would rather say that than publish a roadmap that
only fits the model we started with.

## Roadmap

Ordered by measured cost reduction, not by layer. Two things we learned the hard way shape this
list.

**Rank by cost term, not by software layer.** Cosmos3-Edge measures `p99 = 94.6 ms FIXED +
31.76 ms x NFE`. A stack organised purely by where code lives aims everything at the per-step
term and silently has nothing to offer the one model with a measured deadline problem. So every
pass declares whether it reduces the `FIXED` or the `PER_STEP` term, plus a cost formula, and the
optimizer ranks by `delta_fixed + NFE * delta_step` against the deadline.

**Accuracy-neutral is necessary, not sufficient.** On pi-0's real shapes, swapping eager attention
for SDPA while keeping the mask measures 133.5 to 144-184 us: a regression whose numerics an
equivalence gate would happily certify. Every pass therefore also carries a measured cost delta on
the target's real shapes, and a pass that does not improve its declared term is rejected whatever
its tier.

| | work | attacks | tier |
|---|---|---|---|
| 1 | Paged KV with a device-resident block table; fused write-then-attend | 39.6% of GPU time in gather/copy, and the host syncs behind 51% GPU idle | `BITEXACT` |
| 2 | CUDA-graph capture, gated on a static-shape predicate rather than pipeline position | 469,811 launches per control step at 6.2 us mean | `BEHAVIORAL` |
| 3 | Triton fusion: norm + modulation + QKV + RoPE, and the FFN chain | 18.3% elementwise, 198 launches per layer per forward | `BITEXACT` |
| 4 | Stream overlap and async closed-loop execution | residual idle; converts the deadline from control period to chunk expiry | `BITEXACT` |
| 5 | Guidance branch elision | forwards computing a discarded negative branch | `NUMERIC` |
| 6 | fp8 weights and KV | 17.4% of GPU time in GEMM | `NUMERIC` |
| 7 | Adaptive NFE and velocity-cosine step caching | step count, the only order-of-magnitude lever | `BEHAVIORAL` |

Item 2 is not last. Capture needs static shapes, which for LingBot-VA means item 1 first, but that
precondition is vacuous for models with no paged pool: measured unmodified, capture is worth 4.76x
on GR00T's action head and 4.00x on pi-0's step body. Treating "capture last" as a pipeline
invariant rather than a per-model predicate produced a 28:1 priority inversion in an earlier draft
of this list.

The measurement that explains the whole ordering: 469,811 launches x 6.67 us of CPU enqueue is
3134 ms, against 3031 ms of measured GPU idle. The idle *is* the enqueue. This is a launch
elimination problem, not a kernel tuning problem.

## Getting Started

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

Good first work is in the [Roadmap](#roadmap) below. It is ordered by what each step *unlocks*,
not by what it implements.

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
