# Salvaging PR #2 — four small PRs against the current tree

**PR #2 (`layer4-attention-integration`) will not be merged.** It is kept as a historical branch.
This is the plan that replaces it.

## Why it cannot be merged

It branched on **2026-08-03**, before the One Runtime reorganization. Its 35 files are not in textual
conflict with main — the layout they target no longer exists:

| PR #2 touches | now lives at |
|:--|:--|
| `instinctwm/optimizer/passes/graph_capture.py` | `instinctwm/passes/lingbot/graph_capture.py` |
| `instinctwm/optimizer/passes/hoist_invariant_casts.py` | `instinctwm/passes/lingbot/hoist_invariant_casts.py` |
| `instinctwm/optimizer/released.py` | `instinctwm/verify/released.py` |
| `instinctwm/kernels/*` | `instinctwm/backends/*` |
| `instinctwm/passes/attention_backend.py` | `instinctwm/backends/attention/` — architecture only, nothing wired |
| `instinctwm/adapter/*` | `instinctwm/adapters/*` |

Two of its twelve commits **already landed** during the reorg and must not be re-applied:

- `Fix: the eager fallback in graph capture stopped advancing the KV ring` → shipped as P005 **v1.0.1**
- `Stop the test runner from reporting a real failure as a skip` → in `tests/run_tests.py`

## The rule for all four PRs

> **Re-measure every performance claim. No number collected before the runtime reorganization or
> before the current operating point may be re-landed.**

That is not a formality here. Three specific claims in PR #2 are known-stale:

| claim in PR #2 | why it is stale |
|:--|:--|
| SDPA: flash 0.290 ms → cuDNN 0.195 ms, **1.49×** | measured on **A100-SXM4-80GB**. The current box is **H100**, and the post-P007 kernel census shows attention already dispatching `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16` — the cuDNN kernel. The dispatcher's choice is *different on this GPU*, so the premise of the win is gone. |
| `Take the KV extent out of the capture key: **1.32× per episode**` | measured when graph capture was profitable (P005 shipped at 1.38× on a ~2.5 s cycle). At the current operating point capture measures **503.5 ms against 351.4 ms — 1.43× slower** ([LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md)). |
| residual-fusion rejection numbers | pre-reorg. The *rejection* still stands; the figures need re-dating. |

And the regime model now prices any device-side attention change in advance: the transformer returns
**0.145 ms of cycle per ms of device time** ([LAYER6_REGIMES.md](LAYER6_REGIMES.md)), so eliminating
attention *entirely* would return 6.4 ms. That is why PR 1 below is architecture only.

---

## PR 1 — Layer 4 attention backend wiring (architecture only)

**Scope.** Add `ATTENTION_OP` sites to the adapters and wire `backends/attention/`'s existing
`select()` into the plan, so a site can be enumerated, described and offered to the registry. Port
`instinctwm/passes/attention_backend.py` onto `backends/attention/` and the current pass interface.

**Explicitly out of scope: any performance claim.** No default is changed, no backend is selected on
the serving path, and no latency number appears in the PR. Main already has the architecture
(`backend.py`, `capabilities.py`, `registry.py`, `semantics.py`, `site.py`, `reference.py`); this
completes the wiring so Layer 4 is real rather than declared.

**Gate.** `max |Δ action| = 0` with the pass installed and selection disabled — the pass must be a
no-op until something asks it to choose. Plus the existing `tests/test_attention_backend.py`.

**What re-measurement would be needed if a default ever changed.** Per-backend timings on H100 at the
served shapes, and a NUMERIC certificate: PR #2 measured cuDNN differing from flash by
`max|Δ| = 4.883e-04` on 730,892 of 1,474,560 elements. Note also that the only H100-relevant variant
left is per-shape selection — FLASH split-KV for the 180 `Sq=32` action calls, cuDNN for `Sq=240` —
which the Layer 5 screen priced at **1.5–1.7 ms of cycle**, below the 10 ms bar.

## PR 2 — `IWM_MAX_GRAPHS`

**Scope.** The environment-variable cap on the graph cache, defaulting to 64, plus the finding that
motivated it.

**This knob is a documented negative result, and it should land as one.** Its own experiment
falsified the idea: at a cap of 32, `captures=523 replays=20881 held=32 evicted=461 fallbacks=1`, and
all 8 servers OOMed anyway. Graph eviction does not return its private memory pool, so capping the
*held* set only increases the *eviction* count. One A100 entered `GPU requires reset` and was lost
for the session.

**Why it is still worth landing.** It is an independent, operational reason capture is unshippable at
scale — distinct from the latency reason already recorded. `LAYER5_GRAPH_PERSISTENCE_RESULT.md` says
capture is 1.43× slower; this says it also cannot finish a 50-task evaluation. Both belong in the
record.

**Re-measurement.** None required for the knob itself. The OOM evidence is a *mechanism* claim, not a
latency claim, but it was collected on an 8×A100 fleet pre-reorg and must be dated as such in the PR
body rather than presented as current.

## PR 3 — vLLM-Omni evaluation arm

**Scope.** `omni-arm-requirements.{in,txt}`, `run_omni_arm.sh`, `serve_omni_arm.py`,
`probe_encode_prompt.py`, and the `env.sh` additions. Self-contained under `eval/`; touches no
runtime code.

**Why.** An external comparison arm is the only thing in PR #2 that measures something the project
cannot measure about itself, and it serves the product thesis directly.

**Care required.** It ships an 890-line lockfile and a second interpreter. The existing
constraint holds absolutely: **it must not install flash-attn into `.venv-server`.** The Cosmos3-Edge
work documents why — `/home/ubuntu/iwm_shims/flash_attn` is what stops `diffusers` detecting
flash-attn, switching `autoencoder_kl_wan` to a flash path, and invalidating the 91.6% RoboTwin
baseline. The PR must state which interpreter it uses and assert `.venv-server` is untouched, the way
`eval/cosmos3_edge/README.md` does.

**Re-measurement.** All of it. Any comparison number must be taken on the current box against the
current baseline, and the arm's own environment must be re-locked.

## PR 4 — Operator-fusion pass, with its rejection evidence

**Scope.** `instinctwm/optimizer/passes/operator_fusion.py` → `instinctwm/passes/`,
`instinctwm/runtime/fused_residual.py`, `probe_fused_residual.py`, `tests/test_operator_fusion.py`.

**Land it as rejected infrastructure**, the way the RoPE kernel and the Plan Buffer are held: the pass
exists, defaults off, and its docstring carries the measurement that rejected it.

**Re-measurement.** The rejection needs re-running before the numbers are quoted, and the regime model
predicts the outcome will not change: residual fusion is transformer-side device work at ×0.145, and
the Layer 5 screen found the whole eager-elementwise population — 67.0 ms across 14,495 launches —
worth 9.7 ms of cycle if it vanished completely. A fusion pass recovering a fraction of that cannot
clear the bar. Re-land it to close the question, not to reopen it.

---

## Order and rationale

1. **PR 2 (`IWM_MAX_GRAPHS`)** — smallest, no measurement dependency, purely additive.
2. **PR 1 (attention wiring)** — architecture only, unblocks Layer 4 without claiming anything.
3. **PR 4 (fusion + rejection)** — closes a question; predicted outcome already known.
4. **PR 3 (vLLM-Omni)** — largest surface and the only one that touches environment setup.

## What is deliberately not salvaged

- **The ring-attention kernel** (`kernels/ring_attention.py`, `runtime/ring_attention_install.py`,
  `tests/test_ring_attention.py`) and **"KV extent out of the capture key, 1.32×/episode"**. This is
  the same idea as the Plan Buffer — remove ring state from the graph key — reached through a
  length-parameterized attention kernel instead of a device-resident buffer. The Plan Buffer version
  passed every correctness gate and was **1.43× slower** at the current operating point. Re-landing
  the kernel would mean re-opening a frozen negative result, and it should only happen if someone
  first re-measures capture and finds it profitable again.

  Worth noting the one thing that would justify revisiting it: PR #2 records the `--ring-attention`
  arm running the 50-task workload at a **flat 41–42 GB with zero fallbacks** against the shipped
  default's climb to the 80 GB ceiling. That is a *memory* argument, not a latency one, and it is the
  only surviving reason to care about this branch.

- **The two commits already on main** (P005 v1.0.1 ring fix, run_tests skip-vs-fail).
