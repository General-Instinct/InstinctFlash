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

Good first work is in the [Roadmap](#roadmap) below. It is ordered by what each step *unlocks*,
not by what it implements.

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
