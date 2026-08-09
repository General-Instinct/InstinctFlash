# What ships, what is infrastructure, what is historical evidence

Audit of `main` at `456f91b`, 2026-08-09. Three categories, one rule each:

- **SHIPS** — runtime code with an active, certified effect on served behaviour.
- **INFRASTRUCTURE** — reusable instruments and framework. No effect on serving; used to decide.
- **HISTORICAL EVIDENCE** — kept so a rejected idea is not re-proposed. Must default off and say so.

---

## The finding this audit turned up first

**The registry and the served configuration disagree.** `verify/released.py` lists seven passes as
released with `is_verified() == True`. The certification launch line — identical in `run_pdd_cert.sh`
and `run_sweep.sh` — is:

```
--no-fsdp --no-empty-cache --no-debug-dump --conditioning-prefill --ring-kv
```

That enables **P001, P002, P003 and nothing else.** Every pass in `serve_variant.py` is opt-in
(`action="store_true"`); there is no default-on set. So:

| pass | registry says | flag | in a launch script? |
|:--|:--|:--|:--|
| P001 `substrate_elision` | released 2.110× | `--no-fsdp --no-empty-cache --no-debug-dump` | **yes** |
| P002 `conditioning_prefill` | released 1.120× | `--conditioning-prefill` | **yes** |
| P003 `ring_kv_addressing` | released 1.400× | `--ring-kv` | **yes** |
| P004 `hoist_invariant_casts` | released 1.020× | `--hoist-casts` | no |
| P005 `graph_block_stack` | released 1.380× | `--graph-blocks` | no |
| P006 `stable_state_pools` | released 1.520× | `--stable-pools` | no |
| **P007** `conv_layout_ndhwc` | released **1.405×**, NUMERIC, 555-episode certificate | `--conv-layout` | **no** |

Two consequences, and the second is the one to act on.

**P005 is contradicted in three places at once.** `released.py` calls it released, BITEXACT, 1.380×,
verified. `serve_variant.py`'s own `--graph-blocks` help says *"[NOT SHIPPABLE — 2.17× but NOT
bit-exact, max |Δ action| 1.398 = 136% of real movement … kept for measurement only]"*. And
[LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md) measures capture at
**503.5 ms against 351.4 ms — 1.43× slower** at the current operating point. Its registered 1.380×
was measured on a ~2.5 s cycle that no longer exists. P006 exists only to let P005's graphs survive a
reset ("Only has an effect with `--graph-blocks`"), so it has no served effect either.

**P007 — the only pass that measurably moves the current cycle — is wired into no launch script.**
`--conv-layout` appears in `serve_variant.py`'s flag list and in documentation, and in no `.sh`. The
one certified win is not in the default run. That is the single actionable gap this audit found.

---

## A. What ships

Runtime code, active in the certified launch line, gated at `max |Δ action| = 0`.

| | what it does |
|:--|:--|
| `passes/lingbot/substrate.py` (P001) | removes FSDP-at-world-size-1, per-chunk `empty_cache`, blocking debug dumps |
| `passes/lingbot/conditioning_prefill.py` (P002) | caches episode-constant cross-attention K/V for all 30 layers |
| `passes/lingbot/ring_kv.py` (P003) | interval addressing instead of a boolean mask + `nonzero`; **also what makes the block capturable and compilable at all** |
| `runtime/lingbot_install.py`, `runtime/loader.py`, `runtime/block_heads.py` | the install seam and checkpoint loading |
| `adapters/`, `descriptors/`, `executors/`, `planners/` | the One Runtime spine: capabilities in, no branching on training method |
| `verify/released.py`, `verify/certify.py` | the release registry and the certificate gate |

**Should ship and does not yet: P007 `conv_layout_ndhwc`** — `backends/conv/` plus
`--conv-layout`. NUMERIC, certified on 555 paired episodes (delta −0.0036, exact McNemar p = 0.897,
non-inferiority p = 0.00031), 1.405× measured under ABBA. Add `--conv-layout` to `run_pdd_cert.sh`
and `run_sweep.sh`, or state explicitly why the certified configuration is not the served one.

## B. Infrastructure

Reusable. Decides things; changes nothing about serving.

| | |
|:--|:--|
| `verify/attribution.py` | operator × callsite with a coverage gate (`MIN_COVERAGE` 0.60, `MAX_COVERAGE` 1.10). Validated on a synthetic inverted ranking in `tests/test_attribution.py`, which has caught four instrument bugs |
| `backends/conv/` | the planner → backend → verification flow; legality per **(backend, layout)** pair, tier derived not claimed. P007 is its reference implementation |
| `backends/attention/` | same shape, architecture only — `backend/capabilities/registry/semantics/site/reference`. Nothing wired to a serving site |
| `backends/rope.py`, `backends/torch_fused.py`, `backends/triton_residual.py`, `backends/regions.py` | the kernel-region framework |
| `runtime/state/scratch.py` | scope-bumped arena. Infrastructure, not a shipped pass |
| `tests/perf_gate.py` | correctness unconditional, speed `NOT EVALUATED` on a contended device |
| `tests/test_runtime_boundary.py` | fails if `runtime/` imports a training package or reads provenance |
| **graph-cache telemetry** in `passes/lingbot/graph_capture.py` (`max_graphs`, `captures=/replays=/held=/evicted=`) | already on main; the reusable half of the `IWM_MAX_GRAPHS` experiment |
| `eval/lingbot_va_robotwin/` measurement instruments | `profile_attribution.py`, `probe_device_gaps.py`, `probe_slope_clean.py`, `probe_backend_screen.py`, `probe_p007_passthrough.py`, `compare_arms.py`, `certify_run.py` — all re-runnable, all with their own NOT EVALUATED gates |
| `eval/cosmos3_edge/` | second reference model: `probe_mot_stack.py`, `profile_stack.py`, `probe_compile.py`, `probe_numerics.py` |

## C. Historical evidence

Exists so it is not re-proposed. Every item defaults off and carries its own refutation.

| | verdict |
|:--|:--|
| `passes/lingbot/cfg_elision.py` | **RULED OUT.** Branch 1 live on both axes: 5.64 corrupting its value, 5.39 suppressing only its KV writes, against 1.03 chunk-to-chunk |
| `passes/lingbot/persistent_graph.py` | **NOT SHIPPED.** All correctness gates pass; 503.5 vs 351.4 ms |
| `passes/lingbot/fused_qkv.py` | **NOT SHIPPED.** 1.9% predicted, 0.2% slower; its certificate found M=7 differing in 55 of 64,512 words |
| `passes/lingbot/step_scope_cast.py` | implemented, BITEXACT, 1,740 casts removed, 0.66% — unshipped |
| `passes/lingbot/forward_scratch.py`, `static_partition_hoist.py` | **struck on Cosmos3-Edge**: P8 measures 1.000× on the shipped path (subsumed by capture); P3 obsolete, upstream implemented it |
| `backends/rope.py` (the Triton kernel) | rejected: 1.10× at region scale, 0.3% of cycle |
| `probe_action_terminal*.py` | annotated with their refutation — dead for ~38 cycles, live after the ring wraps |
| `LAYER5_GRAPH_PERSISTENCE_RESULT.md`, `LAYER5_QKV_EXACTNESS.md`, `LAYER5_CAST_FAMILY.md`, `LAYER6.md` §H/§I, `LAYER5_COMPLETE.md` | the written record of each |

**Superseded documents that still read as current** — `LAYER5_NEXT.md` (a ranked proposal whose
ranking term was later refuted), `LAYER5_CRITICAL_PATH.md` §4 (retracted in place, correctly), and
`AUDIT.md`/`PROFILE.md` (pre-Layer-6). These need a header pointing forward, not deletion.

---

## The five dispositions

**1. `IWM_MAX_GRAPHS` — historical evidence, and drop the code change.** The reusable part is
*already on main*: `graph_capture.py` has the `max_graphs` parameter and the
`captures=/replays=/held=/evicted=` counters. PR #2 adds only the environment-variable override —
three lines in `serve_variant.py` — and its own experiment falsified it: at a cap of 32,
`captures=523 replays=20881 held=32 evicted=461 fallbacks=1` and all 8 servers OOMed anyway, because
graph eviction does not return its private pool and capping the *held* set only raises the *eviction*
count. **Revised from the salvage plan: this is not a PR.** It is a paragraph in the graph-persistence
record. The knob would only matter if capture were served, and it is not.

**2. Layer 4 attention wiring — architecture only.** No A100 numbers. flash 0.290 → cuDNN 0.195 ms
(1.49×) was measured on A100; this box is H100 and the post-P007 census shows attention *already*
dispatching `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16`. The dispatcher's choice is
different here, so the premise is gone. Gate: `max |Δ action| = 0` with the pass installed and
selection disabled — a no-op until something asks it to choose.

**3. Operator fusion — rejected infrastructure.** Defaults off, docstring carries the measurement.
Only a **new H100 cycle-level result** changes that, and the regime model says one is unlikely: the
entire eager-elementwise population is 67.0 ms of device time across 14,495 launches, worth **9.7 ms
of cycle** at the transformer's 0.145 exchange rate if it vanished completely. A fusion pass
recovering a fraction of that cannot clear a 10 ms bar. Re-land to close the question.

**4. vLLM-Omni — isolated under `eval/`.** No runtime imports, its own lock and interpreter. The
hard constraint: **it must not install flash-attn into `.venv-server`** — `/home/ubuntu/iwm_shims/flash_attn`
is what stops `diffusers` switching `autoencoder_kl_wan` to a flash path and invalidating the 91.6%
baseline. The PR must assert `.venv-server` is untouched, as `eval/cosmos3_edge/README.md` does.

**5. Ring-attention — record memory separately from latency.** Two independent claims that must not
travel together:

| claim | status |
|:--|:--|
| **latency** — KV extent out of the capture key, 1.32×/episode | **stale.** Measured when capture paid 1.38× on a ~2.5 s cycle. Capture now measures 1.43× slower. Do not re-land |
| **memory / stability** — the `--ring-attention` arm held a flat **41–42 GB with zero fallbacks** over a 50-task run, where the shipped default climbed from 24 GB to the 80 GB ceiling and OOMed all 8 servers, one A100 entering `GPU requires reset` | **independent of latency and still unrefuted.** This is the only surviving reason to care about that branch |

The memory result belongs in the graph-persistence record as a *second, independent* reason capture is
unshippable — distinct from the latency reason. It should not be cited as support for the 1.32×.

---

## Recommended actions, smallest first

1. **Wire `--conv-layout` into `run_pdd_cert.sh` and `run_sweep.sh`**, or record why the certified
   configuration is not the served one. This is the only gap where something certified is not running.
2. **Reconcile P005/P006 in `released.py`** with the `--graph-blocks` help text and the 1.43× result.
   Their registered speedups describe an operating point that no longer exists. A frozen pass changes
   only for a correctness bug — a registry that contradicts the measured runtime is that bug.
3. **Add forward-pointing headers** to `LAYER5_NEXT.md`, `AUDIT.md`, `PROFILE.md`.
4. **Drop item 2 from [SALVAGE_PR2.md](SALVAGE_PR2.md)** — `IWM_MAX_GRAPHS` is documentation, not a PR.
