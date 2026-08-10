# External adoption audit

Two outside positions, audited 2026-08-10: someone who only wants to run a checkpoint, and someone
who wants to add a model family. LeRobot and vLLM are the product references. Everything below is
either a measurement or a citation to a public artifact.

---

## 1. External user verdict: **NOT READY**

One blocker, and it is not a documentation problem.

**A robot policy that cannot be called twice is not usable.** Measured cold on the real checkpoint:

| stage | time | note |
|:--|--:|:--|
| `describe()` | 0.29 s | one metadata file, no weights |
| `from_pretrained()` | 16.41 s | **20.36 GB** fetched |
| `reset()` | 9.17 s | |
| first `predict()` | 2.54 s | **first action in 28.4 s** |
| closed loop | **1 / 4** | breaks at cycle 2 |

The mechanism to loop now exists — `predict` drives an optional `commit(observation, action)` hook —
and the external toy model uses it successfully. LingBot-VA's adapter does not implement it yet, so
cycle 2 dies inside the model on a missing temporal tap. This is a bounded piece of adapter work, not
a design question, and it is the only thing between here and READY on the code side.

### Fixed during the audit

- `huggingface_hub` was in **no** dependency list, so the documented install produced a runtime that
  could not resolve the repo id the README opens by loading. Now a core dependency.
- The README taught `Optimizer` / `Tier` / `plan.serve(model, port=29056)` as the Quick Start —
  planner and transport vocabulary in the first example — and mentioned `Runtime.from_pretrained`
  only as a *"Next milestone… proposal"*. The shipped API now opens the README.
- No hardware requirement was stated anywhere; users met OOM instead. Now stated (~30 GB free).
- `requirements-serving.txt` was referenced in no user-facing document.
- The README's example Hub id `example-org/wm-blockheads-2v4a` returns 401; it is a local path.

### Still open, and owned by you

- **`general-instinct/lingbot-va` returns 401.** The repo does not exist publicly. The token
  available here is scoped to a personal namespace (`roleInOrg: None`), so I cannot create it. Until
  it exists, the documented first line fails for everyone.
- **Not on PyPI**, so the model card's `pip install instinctwm` fails. The README now says so
  explicitly rather than letting users discover it.
- **License contradiction**: `pyproject.toml` declares `AGPL-3.0-or-later`; the generated model card
  declares `license: apache-2.0`. Both cannot be right, and AGPL on a serving runtime is itself an
  adoption blocker in industry — LeRobot is Apache-2.0. This is a decision, not a bug fix.
- **`LINGBOT_ROOT`** requires a separate upstream checkout that no public document mentions. For an
  outside user this is unguessable; vendoring the serving shim would remove it.
- 23 root-level markdown files (`LAYER5_QKV_FEASIBILITY.md`, `SALVAGE_PR2.md`) present a research
  notebook rather than a product.

---

## 2. External model author verdict: **PASS**

A new model family integrates with **no PR to InstinctWM**. Demonstrated, not asserted:
[`examples/external_plugin/`](examples/external_plugin/) is a working integration for a toy
autoregressive model chosen to be structurally unlike LingBot-VA — one phase, one forward, no
streams, no guidance, no commit, no frozen components.

| measure | result |
|:--|:--|
| required adapter methods | **2** — `spec()`, `build_in_process()` |
| optional impl methods | `reset`, `commit`, `close` |
| model-specific lines | **80** adapter + 45 model |
| required checkpoint metadata | `model_id`, `backbone`, `servable` |
| capability declarations | none required; declaring nothing offers nothing |
| planner knowledge required | **none** |
| InstinctWM internals to understand | `AdapterSpec`, `PhaseSpec`, and the impl verbs |
| registration | one entry point in the author's own `pyproject.toml` |

Model knowledge (`vocab`, `dim`, `history`) stays in the author's `config.json` and adapter;
`instinctwm.json` carries execution facts only. That separation held under a genuinely different
model, which is the strongest evidence the declaration is not secretly LingBot-shaped.

### What the audit had to fix to get here

1. `InProcessBackend.predict` called `impl.infer(dict)` — LingBot's **server** verb. An adapter
   implementing the obvious `predict` got `AttributeError` from inside the runtime.
2. A checkpoint could not bootstrap its own adapter: `register()` only fires if something already
   imported the plugin. Added entry-point discovery under `instinctwm.adapters`.
3. **The planner was lying to every non-LingBot model.** `Optimizer` defaults to LingBot's pass list,
   and three passes that patch the LingBot server object declared no applicability — so a toy GRU
   received a plan reporting `APPLY fsdp_elision` with *"expected: measured 1.75x standalone on
   LingBot-VA."* The gate already existed; those passes never used it.

### Still rough

`BackendAdapter` in `adapters/base.py` still documents `spec` / `install` / `serve(plan, port)` —
a contract the facade does not use, and one that puts a **port** in the extension surface. The real
contract is `spec` + `build_in_process`. That divergence should be resolved in favour of the real one.

---

## 3. The recommended public Runtime API

```python
from instinctwm import Runtime, describe

describe("org/model")                                  # metadata, no weights

runtime = Runtime.from_pretrained("org/model")          # model lifetime
with runtime.episode(prompt="...") as episode:          # episode lifetime
    while not done:
        action = episode.predict(observation)           # one control cycle
runtime.close()
```

Three lifetimes, three objects, and nothing else public.

| lifetime | expressed by | why |
|:--|:--|:--|
| model | `Runtime`, context manager, `close()` | expensive, GPU-resident, shared |
| episode | `Runtime.episode()` → `Episode`; `reset()` for the simple case | must be a value once a fleet shares one model |
| control cycle | `predict(observation, executed_action=None)` | the only verb a user needs |
| internal phases | **private** | not a user concern |

**Decisions, with reasons.**

- **`reset()` — keep**, alongside `episode()`. It is LeRobot's shape and correct for one robot in one
  loop. It cannot express "these two rollouts are separate", which is why `episode()` exists too.
- **`predict()` — keep, and it must loop.** Absorbing phases here is right *only* because the phase
  is genuinely internal. The part that is **not** internal — what the robot actually executed, which
  a safety filter may have changed — is surfaced as `executed_action=`.
- **`step()` — reject.** Ambiguous between a denoising step and a control step, and it buys nothing.
- **`commit()` — reject as public.** It is a phase. Models declare it to the *runtime*, never to the
  user. Exposing it would put LingBot's ring in every future model's API.
- **`close()` and context managers — keep** on both lifetimes.
- **WebSocket, worker placement, KV state, ring state — stay private.** Placement is chosen at load
  from a `placement=` hint and never appears in the model abstraction.

Designed for 2030: a stateless diffusion policy declares no streams and no commit and works; an
autoregressive world model declares a commit and works; neither needs a change here.

---

## 4. Compared with LeRobot and vLLM

**Adopt.** LeRobot's single-entry Hub UX (`from_pretrained` on a repo id, nothing else to know) and
its habit of generating model cards from the artifact so they cannot drift. vLLM's insistence that
cache and scheduling state are never in the user's API, and its request-id notion of a session —
which is what `Episode` is.

**Do not adopt.** LeRobot's model-class-per-policy dispatch: `get_policy_class(cfg.type)` means a new
family needs a class inside LeRobot, i.e. a PR. Entry-point discovery is strictly better for an
ecosystem. Also not adopting vLLM's batch-first `generate()` surface — robot control is a closed loop
over a live world, not a batch of prompts, and pretending otherwise would distort the API.

**Keep deliberately different — and this is the real differentiator.** The chain
`checkpoint → declaration → capabilities → planner → passes → runtime` has no equivalent in either
project. LeRobot loads a *class*; InstinctWM loads a *declaration*, and the runtime derives what is
legal. `publishability()` — "can I ship these weights without shipping my recipe?" — is a genuinely
novel product primitive. The two-namespace split (`execution` the runtime may read, `provenance` it
provably cannot) is worth more than API parity and should not be traded for it.

---

## 5. Smallest prioritized changes to make both verdicts PASS

Ranked by adoption impact. Only the first four are blockers.

1. **Implement `commit()` in the LingBot-VA adapter** so `predict()` loops. Without it the flagship
   model is not usable. *Engineering, bounded, needs the model's keyframe semantics.*
2. **Publish `general-instinct/lingbot-va`** — needs org write on the token. *Yours.*
3. **Publish `instinctwm` to PyPI**, so the model card's install line is true. *Yours.*
4. **Resolve the AGPL / Apache-2.0 contradiction**, and relicense if industrial adoption is the
   goal. *Yours — a licensing decision, not a fix.*
5. Remove or vendor `LINGBOT_ROOT`, so no outside user needs a second checkout.
6. Reconcile `BackendAdapter` with the contract the facade actually calls; drop `port` from it.
7. Move the ~20 `LAYER*` / `SALVAGE` documents under `docs/`, leaving a product-shaped root.

Items 2–4 are yours and are minutes of work. Item 1 is the only real engineering, and it is the one
that decides whether this is a product or a benchmark harness.
