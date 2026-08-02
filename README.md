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

**Accuracy-neutral is necessary, not sufficient.** Measured on LingBot-VA: `HoistInvariant` is
bit-exact and costs **+133 ms per cycle under eager execution** while paying for itself under graph
capture. A pass therefore carries a measured cost delta *per executor*, and correctness does not
imply admission — a pass with no measurement on the target executor is not admitted, because
"bit-exact" says nothing about whether it helps there.

| | work | attacks | tier |
|---|---|---|---|
| 1 | Paged KV with a device-resident block table; fused write-then-attend | 39.6% of GPU time in gather/copy, and the host syncs behind 51% GPU idle | `BITEXACT` |
| 2 | CUDA-graph capture, gated on a static-shape predicate rather than pipeline position | pre-capture: ~250k dispatches/cycle at 6.2 us mean. **Done** — 1.21x whole-episode, bit-exact, but the key does not converge (~6 captures/cycle indefinitely) | `BITEXACT` |
| 3 | Triton fusion: norm + modulation + QKV + RoPE, and the FFN chain | 18.3% elementwise, 198 launches per layer per forward | `BITEXACT` |
| 4 | Stream overlap and async closed-loop execution | residual idle; converts the deadline from control period to chunk expiry | `BITEXACT` |
| ~~5~~ | ~~Guidance branch elision~~ **RULED OUT on this backend** | a two-axis liveness test found the action stream's CFG branch 1 live on both axes: corrupting its return moved the result 5.64, suppressing only its shared-KV writes moved it 5.39, against 1.03 chunk-to-chunk movement. Its output is discarded but the computation is load-bearing through shared state | — |
| 6 | fp8 weights and KV | 17.4% of GPU time in GEMM | `NUMERIC` |
| 7 | Adaptive NFE and velocity-cosine step caching | step count, the only order-of-magnitude lever | `BEHAVIORAL` |

Item 2 is not last: capture needs static shapes, which for LingBot-VA means item 1 first, but that
precondition is a per-model predicate rather than a pipeline invariant.

The measurement that set this ordering: per-op cost is **6.2 us, of which 83.6% is
`cudaLaunchKernel` itself**, and an `add_` costs the same on 1 element as on 1 MB. It was a launch
elimination problem.

**That is no longer the binding constraint.** With the launches inside captured graphs, GPU time
binds again, and the ordering above is stale — see the current ranking, which is derived from a
re-profile rather than from this list. Two things it now gets wrong: guidance elision is ruled out
(above), and attention backend work is worth 7% of GPU busy, not a priority.

Capture also does **not** converge on this model. The KV ring advances 152 slots/cycle and the read
extent grows every cycle, so the graph key moves every cycle and ~6 graphs are captured per cycle
indefinitely. Capture is still worth 1.21x whole-episode; it is not free, and whole-cycle capture is
blocked structurally rather than pending.

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
