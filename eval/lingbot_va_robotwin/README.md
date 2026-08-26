# LingBot-VA × RoboTwin 2.0 — evaluation pipeline

The accuracy reference for InstinctFlash. Every future optimization (step distillation, CUDA
graph capture, kernel work, KV-cache changes) is judged against numbers produced here, so
this directory optimizes for *being correct and auditable*, not for being fast to run.

## Shipped configuration

`shipped_configuration()` is the source of truth for the production LingBot serving flags:

```text
--no-fsdp --no-empty-cache --no-debug-dump --conditioning-prefill --ring-kv --conv-layout
```

This serves P001 (substrate elision), P002 (conditioning prefill), P003 (ring KV addressing), and
P007 (convolution layout) at an overall NUMERIC tier. P005 (`--graph-blocks`) and P006
(`--stable-pools`) remain available for measurement but are **NOT RECOMMENDED** in the shipped
configuration.

## What it is

[LingBot-VA](https://github.com/robbyant/lingbot-va) is an autoregressive video-action
world model: a dual-stream Mixture-of-Transformers on a Wan2.2 backbone that predicts
future video latents and action chunks in one interleaved sequence. It reports **92.9%
easy / 91.6% hard** on RoboTwin 2.0's 50 tasks, and ships a posttrained RoboTwin
checkpoint — so unlike the earlier Cosmos3-Edge work, **no SFT is needed and there is no
embodiment mismatch**: the checkpoint's 16 used action channels are exactly RoboTwin's
`take_action(a, 'ee')` layout.

Two processes, two Python environments, one websocket:

```
  RoboTwin/.venv  (sapien 3.0.0b1, torch 2.4)          .venv-lingbot  (torch 2.9, diffusers 0.36)
  ┌──────────────────────────────────┐   ws://:2905N   ┌──────────────────────────────────┐
  │ eval_polict_client_openpi.py     │ ──────────────▶ │ wan_va_server.py                 │
  │  simulator, expert-gate, scoring │ ◀────────────── │  T5 + Wan VAE + 5B MoT           │
  └──────────────────────────────────┘                 └──────────────────────────────────┘
```

The two environments are dependency-incompatible on purpose; the websocket is the seam.
That seam is also InstinctFlash's abstraction boundary — swapping the model under test should
be a change of what listens on the port, nothing else.

## Wire protocol (server reads exactly 5 keys)

| message | keys | server does |
|---|---|---|
| reset | `reset=True`, `prompt` | clears transformer KV cache + VAE cache, runs T5 on the instruction once |
| kv | `compute_kv_cache=True`, `obs=[frames]`, `state=action` | VAE-encodes real frames, folds observed history into the KV cache |
| infer | `obs` | 25+1 video denoise steps, then 50+1 action denoise steps → 32 control actions |

The server is **stateful per episode**. One client per server, always.

### The two-phase episode contract (what `Runtime.predict()` hides)

One upstream control cycle is TWO calls: `infer(obs)` produces the action chunk, then a second
`infer(compute_kv_cache=True, state=<executed action>, obs=<frames observed while it executed>)`
advances the KV ring. The two calls want DIFFERENT observations — the action forward conditions on
the current frame only (the first chunk of an episode on exactly ONE frame, streaming VAE), while
the ring advance consumes the whole window: 4 frames on the first advance, 8 after. Skip the
commit and the ring never moves; the model fails with a conv-size error several chunks later.
InstinctFlash's `Runtime.predict()` folds all of this in — it holds the returned action, folds it
into the ring on your NEXT `predict()` together with the frames observed while it executed, and
raises a readable error when the window is short — so a caller just loops
`runtime.predict({"obs": frames_since_last_call, "prompt": task})`. Only clients speaking the raw
wire protocol above (stock `wan_va_server`, `serve_variant.py`) issue the two calls themselves.

## Running it

```bash
cd eval/lingbot_va_robotwin
source ./env.sh

IFL_FA_SHIM=1 ./servers.sh start 8      # one server per GPU; refuses to start on a busy port
./servers.sh status

# gate: serving must reproduce the training text conditioning (see below)
$IFL_SERVER_PY check_prompt_parity.py \
  --latent  /home/ubuntu/iwm_parity/.../episode_000000_0_139.pth \
  --empty-emb /home/ubuntu/iwm_parity/empty_emb.pt

./run_eval.sh myrun 100 adjust_bottle place_dual_shoes ...   # fans tasks over the 8 GPUs
$IFL_CLIENT_PY aggregate.py $IFL_RESULT_DIR/myrun --expect-episodes 100 --expect-tasks 50
```

## The traps

Each of these is a real defect found by reading both sides of the wire. They are listed
because *most of them fail silently* — the run completes and prints a plausible number.

**Prompt parity was never checked upstream, and it is the whole ballgame.** LingBot-VA
never runs T5 during training: `lerobot_latent_dataset.py:246` reads a *precomputed*
`text_emb` from each dataset `.pth`, and substitutes a precomputed `empty_emb.pt` for the
CFG drop. The server does the opposite — `wan_va_server.py:421-435` runs T5 live with
`prompt_clean()` and pads to 512. Nothing asserts these agree. This is the exact shape of
a failure that voided a 22.7-hour run on this box. `check_prompt_parity.py` closes it, and
on our checkpoint it passes **bit-exactly** (`max|Δ| = 0.000e+00`, cosine 1.000000 on both
the positive prompt and the CFG-negative branch). Re-run it whenever the checkpoint moves.
Note `encode_prompt`'s own default is `max_sequence_length=226` and only the `_reset` call
site passes 512 — one edit away from a silently different embedding.

**Port collision between the rendezvous and the websocket.** Upstream uses
`START_PORT=29056` and `MASTER_PORT=29061`; at 8 GPUs, GPU 0's torchrun rendezvous port is
GPU 5's websocket port. Seven servers come up, one dies, and the fleet looks healthy. It is
worse than a crash because `WebsocketClientPolicy._wait_for_server` retries on *any*
exception every 5s forever — a client aimed at the dead port hangs indefinitely, which is
indistinguishable from a slow task. `env.sh` keeps the ranges far apart (ws 29056+i, rdzv
29800+i) and `servers.sh` refuses to launch on a busy port and insists on 8/8.

**A crashed task raises the headline.** `calc_stat.py` scores a task with no mp4s as
`None` and then averages only non-`None` rates, dropping it from the denominator instead of
counting it. `aggregate.py` treats a zero-episode task as a hard error and cross-checks the
mp4 evidence against the independently written `res.json`.

**Results are labelled with the wrong policy.** Upstream passes `policy_name=ACT` purely to
locate `policy/ACT/deploy_policy.yml` (a config skeleton — nothing imports an ACT model),
with the side effect that output lands in `eval_result/<task>/ACT/` and is indistinguishable
from a genuine ACT baseline. We keep the skeleton but label the run `LingBotVA`.

**Duplicate tasks overwrite each other.** Success is counted from mp4 filenames
`<test_num>_<prompt>_<True|False>.mp4` and `test_num` restarts at 0 per client process.
Upstream's `launch_client_multigpus.sh` group 6 runs `place_empty_cup` and
`blocks_ranking_rgb` four times each into one `save_root`. `run_eval.sh` refuses duplicates.

**Dead client flags.** Everything after `--overrides` is swallowed by `argparse.REMAINDER`,
so the top-level `--port`/`--test_num` defaults never apply. Worse,
`--video_guidance_scale` / `--action_guidance_scale` *are* parsed by the client and sent on
the wire, but the server never reads them (`wan_va_server.py:379,513,552` use only
`job_config`). Guidance is a server-side config value; the client flag is decoration.

**`infer_mode` is an import-order accident.** `va_robotwin_cfg.py` never sets it. It is
`'server'` only because `va_franka_cfg.py:9` mutates the shared config object after copying
it, before `configs/__init__.py` imports the robotwin config. Reordering that file would
route `--config-name robotwin` into offline video generation and produce no eval at all.

**flash-attn is a hard import that is never called.** `model.py:29-32` imports
`flash_attn_func` at module scope, but both the checkpoint config and the server select
`attn_mode='torch'` (`custom_sdpa`). `IFL_FA_SHIM=1` supplies an import-only stub that
**raises if called**, turning "flash-attn is unused" from an assumption into an enforced
invariant. Drop it once a real wheel is installed — `PYTHONPATH` precedes site-packages and
would otherwise shadow the real package.

## Protocol deviations to state whenever quoting a number

- **Seeds.** The LingBot-VA client uses `st_seed = 10000*(1+seed)`
  (`eval_polict_client_openpi.py:395`); RoboTwin's canonical harness uses
  `100000*(1+seed)`. Both are disjoint from the training seeds (low hundreds), so neither is
  contaminated — but they are *different scene sets*. Numbers from here are comparable to
  the LingBot-VA paper, not to a canonical-seed RoboTwin baseline.
- **Instructions.** `instruction_type` is hardcoded `'seen'` (line 308), ignoring the config.
  A seen-instruction eval is easier than unseen.
- **Latency is not scored.** `take_action` blocks until the policy returns, so a 77-forward
  model ties a 4-forward model. Any latency claim needs its own protocol.
- **Long-horizon tasks.** The KV pool saturates at roughly 36 cycles (~1152 control steps).
  Tasks with `step_lim ≥ 1200` (`blocks_ranking_rgb` 1200, `open_microwave` 1500,
  `put_bottles_dustbin` 1700) can evict real history mid-episode. Report them separately
  until cache occupancy is instrumented.

## Files

| file | role |
|---|---|
| `env.sh` | single source of truth: repos, checkpoint, interpreters, port plan |
| `servers.sh` | fleet lifecycle with preflight port checks and 8/8 verification |
| `check_prompt_parity.py` | **the gate** — live T5 vs baked training embedding |
| `run_eval.sh` | fan tasks over GPUs, one serial worker per GPU, records provenance |
| `aggregate.py` | honest scoring; prints `REPORTABLE: NO` rather than a partial number |

---

# First measured results — LingBot-VA on RoboTwin 2.0

Box: 8× H100 80GB, driver CUDA 13.0, torch 2.9.0+cu126.
Checkpoint: `robbyant/lingbot-va-posttrain-robotwin`.
Date: 2026-07-31 / 2026-08-01.

## 1. Correctness gate

`check_prompt_parity.py` — the server's live T5 output vs the embedding baked into the
training dataset. This is the check whose absence voided a previous 22.7-hour run.

| comparison | shape | max abs Δ | cosine (non-pad) |
|---|---|---|---|
| positive prompt: live T5 vs dataset `text_emb` | (512, 4096) | **0.000e+00** | 1.000000 |
| CFG negative `""`: live T5 vs `empty_emb.pt` | (512, 4096) | **0.000e+00** | 1.000000 |

Serving reproduces the training text conditioning exactly. Checkpoint also loads with
**zero missing tensors** (key-set diff against the model's `named_parameters`); the two
`patch_embedding.*` keys the loader warns about are a vestigial conv the architecture
replaced with `patch_embedding_mlp` (`model.py:575`, old name commented out).

## 2. Accuracy — canonical 50-task baseline

`baseline50`: all 50 official RoboTwin 2.0 tasks x 50 episodes, **stock** server, LingBot-VA
client protocol (`st_seed = 10000*(1+seed)`, `instruction_type='seen'`), `demo_clean`.
50/50 tasks, 2500/2500 episodes, **zero failures**, `aggregate.py` prints `REPORTABLE: YES`.

| metric | value |
|---|---|
| **MACRO mean over 50 tasks** (leaderboard definition) | **91.6%** |
| MICRO pooled | 91.6% (2291/2500), 95% CI [90.5, 92.7] |
| LingBot-VA published | 92.9% easy / 91.6% hard |

14 tasks at 100%. Floor: `hanging_mug` 30.0%, `open_microwave` 60.0%, `turn_switch` 62.0%,
`put_bottles_dustbin` 70.0%, `move_stapler_pad` 70.0%.

Read honestly: `demo_clean` is the *easy* setting, so the comparison is against their 92.9%, and
our 91.6% sits just below the bottom of that comparison (92.9 is outside our [90.5, 92.7]). This
is a close reproduction, not an exact one. Candidate explanations, none yet tested: 50 episodes
per task rather than 100, a different accepted-seed set, or a genuine small deficit. It is a
sound *reference* either way — every optimization is compared against this run, on these pinned
seeds, not against the paper.

Pinned accepted-seed lists and per-chunk action streams are retained internally, so a later arm
can replay the identical scene set and be compared per-episode (contact the maintainers for the
artifacts).

## 3. Latency cost model (batch 1, idle H100)

`probe_latency.py`, 14 cycles, real episode message order. One cycle = 32 control steps =
77 transformer forwards (26 video denoise + 51 action denoise), each at batch 2 because
`guidance_scale=5` forces CFG batch duplication.

| stage | mean | share |
|---|---|---|
| `reset` (T5 encode, once per episode) | 67 ms | — |
| `compute_kv_cache` | 453 ms | 5% |
| `infer` (denoise loop) | 8429 ms | **95%** |
| **full cycle** | **8881 ms** | 277 ms/control step → **3.6 Hz** |

Per-forward: 8429/77 = **109 ms**. The model is ~5.1B params = 10.2 GB bf16; at H100 HBM
3.35 TB/s a memory-bound forward is ~3 ms. **We are ~36× off roofline** — the cost is
overhead, not arithmetic.

Latency also is *not flat*: +7.3% from the first 7 cycles to the last 7, because attention
runs over a KV population that grows 272 tokens per cycle toward a 9792-slot pool
(saturating around cycle 36). Under 8-way GPU sharing the same cycle costs ~16 s (~2 Hz).

## 4. First optimizations — 1.92×, bit-exact

Three costs, all found by reading the code, all removed via `serve_variant.py` (runtime
patches; the upstream tree is untouched). Measured back-to-back on idle GPUs, reference
and variant both under `--deterministic-seed 1234` — required, because `_infer` draws
`torch.randn` unseeded (`wan_va_server.py:449-462`), so two *stock* servers already disagree.

| variant | cycle (mean) | vs stock | control rate | max abs Δ action |
|---|---|---|---|---|
| stock | 8881 ms | 1.00× | 3.6 Hz | — |
| `--no-fsdp` | 5078 ms | **1.75×** | 6.3 Hz | **0.000e+00** |
| `+ --no-empty-cache --no-debug-dump` | 4624 ms | **1.92×** | 6.9 Hz | **0.000e+00** |

Yardstick: the reference model's own chunk-to-chunk action movement is 1.03, so a delta of
exactly zero is not a small number relative to noise — it is *no change at all*.

What each removes:

- **`--no-fsdp`** (the big one). `distributed/util.py:15-19` applies FSDP `fully_shard`
  whenever `dist.is_initialized()`, which is always: `init_distributed` runs
  unconditionally even for a single-GPU server. `fsdp.py:28-34` wraps 4 units per block
  (attn1, attn2, ffn, block) × 30 blocks + root = **121 units** with
  `reshard_after_forward=True`. At world_size=1 every all-gather is a no-op collective, but
  PyTorch still pays the flat-param copy and stream sync per unit per forward — about
  **9,300 shard/unshard round trips per cycle** to shard a model across one GPU.
- **`--no-empty-cache`**. `torch.cuda.empty_cache()` on every chunk
  (`wan_va_server.py:569`) and every KV update (`:603`), returning the caching allocator's
  pool to the driver so the next cycle re-runs `cudaMalloc`.
- **`--no-debug-dump`**. `save_async` (`utils.py:56-70`) is async only for the *disk write*;
  the `.cpu()` at :63-64 is a blocking device→host copy of the full latent and action
  tensors, three times per cycle, unconditional, with no upstream flag. Visible mostly in
  `compute_kv_cache`: 453 → 292 ms.

**These are bit-exact and therefore free.** They need no paired non-inferiority run — which
matters, since establishing non-inferiority costs roughly 10× more GPU time than measuring
a speedup, and a previous project found CUDA graph capture was *not* bit-exact and had to
pay exactly that cost.

## Appendix — flash-attn and the baseline environment

`baseline50` (section 2) ran with the import-only flash-attn shim and NO real flash-attn in the
venv. That matters: with real flash-attn installed, `diffusers` detects it via package metadata
and routes `autoencoder_kl_wan` down a flash-attention path instead of SDPA, which changes the
VAE's numerics. The transformer is unaffected either way (`attn_mode='torch'` → `custom_sdpa`;
`flash_attn_func` is never called).

A real `flash-attn 2.8.3` wheel finished building mid-session and was **uninstalled** so the
environment continues to match the one that produced the 91.6% baseline. Any run whose numbers
are compared against that baseline must keep it uninstalled. If flash-attn is ever wanted for
the VAE, the baseline has to be re-measured first — it is not a free environment change.

## 6. Where the time actually is (measured 2026-08-01)

`conditioning_prefill` — caching the episode-constant cross-attention K/V for all 30 layers
(`model.py:331` withholds `attn_caches` from cross-attention, so the text K/V is re-projected on
all 77 forwards) — removes **89 of 226 TFLOP per control cycle, 39% of all arithmetic**, and is
**bit-exact** (`max|delta action| = 0.000e+00` over 6 paired seeded cycles).

It bought **1.05x**.

| arm (all on top of the 1.92x bit-exact base) | cycle mean | control rate |
|---|---|---|
| base: fsdp + empty_cache + debug dumps elided | 4320.6 ms | 7.4 Hz |
| base + conditioning_prefill | 4115.4 ms | **7.8 Hz** |

That gap is the most useful measurement so far, because it decomposes the remaining cost:

| component of the 3828 ms denoise | ms | share |
|---|---|---|
| arithmetic, 226 TFLOP at ~350 TFLOPS achieved | 646 | 17% |
| weight traffic, 77 forwards x 10.2 GB at 3.35 TB/s | 234 | 6% |
| **launches, host syncs, KV gather** | **~2950** | **77%** |

The pass performed exactly as predicted *on the arithmetic it removes* (predicted ~254 ms,
measured 201 ms). Arithmetic simply is not where the time goes. Two consequences:

1. **Quantization is deprioritized.** It attacks bytes and FLOPs — 23% of the problem.
2. **Sync/gather elimination and CUDA-graph capture move to the front.** They attack the 77%.

Corroborating evidence from the same run: the per-cycle ramp as the KV pool fills grew from
**+7.3%** (stock) to **+30.2%** (optimized) over 12 cycles — 3710 -> 4829 ms. As the fixed
overheads come off, the KV-dependent term dominates, and it keeps growing until the pool
saturates around cycle 36. `model.py:452-453` re-gathers the entire valid KV pool into a fresh
tensor per layer per forward (~240 MB/layer at saturation). That is now the prime suspect and
the next pass.

## 7. ring_kv_addressing — 1.43x, bit-exact (2026-08-01)

First optimization done under the profile -> implement -> benchmark -> generalize cycle, and the
first one aimed by a profile rather than by reading code.

**Profile said:** gather/copy is 39.6% of GPU time and `aten::nonzero` fires 30 x 77 x 3 times per
`_infer`, each a data-dependent shape and therefore a host round trip. Two lines cause both
(`model.py:451-453`): `valid = mask.nonzero(...)` then `key_pool[:, valid]`.

**Implemented:** the boolean mask was only ever encoding an interval. `allocate_slots` already
takes the lowest free slots and evicts the oldest, so allocation is sequential-with-wraparound and
the live set is always a ring interval. Track it as two host ints and `valid` becomes a *slice* —
a view, not a copy. Fast path for contiguous intervals; falls back to stock when the interval
wraps, so keys stay in ascending slot order and the pass stays bit-exact.

**Benchmarked** (14 cycles, idle H100, both arms seeded, everything else held equal):

| arm | cycle | rate | max abs delta |
|---|---|---|---|
| reference | 4277.3 ms | 7.5 Hz | — |
| + ring_kv | **2982.3 ms** | **10.7 Hz** | **0.000e+00** |

**1.43x, bit-exact.** Mechanism confirmed by re-profiling rather than assumed:

| kernel, one `_infer` | before | after |
|---|---|---|
| `aten::nonzero` | 7,034 (65.3 ms) | 104 (0.0 ms) |
| `aten::index` | 6,982 (168.1 ms) | 52 (0.1 ms) |
| `aten::_index_put_impl_` | 13,852 (45.9 ms) | 52 (0.2 ms) |
| total launches | 361,565 | 250,520 |
| GPU time | 2402 ms | 1812 ms |

`aten::copy_` is unchanged at ~48k. Those are the unfused transformer-block temporaries, which is
the next pass, and the profile said so before this one was written.

**Cumulative: 8881 -> 2982 ms, 2.98x, 3.6 -> 10.7 Hz, every step bit-exact.**

Two honest caveats. The pool holds 9792 slots and grows 272 tokens/cycle, so it does not wrap
until roughly cycle 36; a 14-cycle benchmark measures the fast path throughout and therefore
*understates* the steady-state win, since the removed gather grows with occupancy. And the fast
path does not maintain the `mask`/`id`/`is_pred` arrays, so the wrapped fallback currently reads
stale bookkeeping — correct within a 36-cycle episode, wrong beyond it. Fixing that is the first
follow-up, and it is a correctness bug, not a tuning issue.

**What the correctness gate caught.** The first implementation set `pred = key_size` on a
provisional commit. But `update_cache=1` fires *twice* per cycle (video last step at
`wan_va_server.py:504`, action last step at `:544`) and stock `clear_pred_cache` drops every slot
with `is_pred` set, i.e. both blocks. Overwriting instead of accumulating leaked the video block
into the permanent cache. The gate reported `max|delta| = 1.22` against a chunk-to-chunk movement
of 1.03 — larger than the signal, so unmistakably semantic rather than numeric. Two lines to fix.
Without a bit-exactness gate this would have shipped as "1.43x with a small accuracy change".

### 7b. Wraparound correctness, and the model that was wrong

The first version fell back to stock code when the ring interval wrapped, but the fast path did not
maintain `mask`/`id`/`is_pred`, so the fallback would have read stale bookkeeping. Correct within a
36-cycle episode, wrong beyond it -- and RoboTwin episodes run 400-1700 steps, i.e. 12-53 cycles.

Investigating it falsified the model. `tests/test_ring_allocator.py` exercises the REAL
`WanAttention` allocator (constructed via `__new__`, no weights, so it crosses many wraps in
seconds) and shows allocations are always a contiguous run, but the live set goes NON-CONTIGUOUS
161 times in 600 allocations: after `clear_pred_cache` it is `live=9520, lo=0, hi=9791` -- the pool
minus a hole in the middle. Stock presents that in ASCENDING SLOT INDEX order, which a
chronological ring does not reproduce, and reordering keys changes the floating-point reduction.

Rewritten so that the live set is read as slice-or-cat in ascending order, and `mask`/`id`/`is_pred`
are maintained stock-exact BY SLICE -- predicted from the tracked interval rather than found with
`nonzero`. There is no path left on which a stale mask can be observed.

| gate | result |
|---|---|
| allocator parity vs stock, 200 cycles (~5.6 full wraps) | **800/800 checks, 0 mismatches**, identical indices in identical order |
| bit-exactness, 40 cycles (past the wrap at ~36) | `max abs delta = 0.000e+00` |
| long-horizon `put_bottles_dustbin`, 1700 steps ~= 53 cycles/episode, 3 pinned seeds | **3/3 bitwise-identical action streams**, outcomes 2/3 on both arms, McNemar p = 1.0 |

Performance improved on the corrected version:

| arm | cycle | rate | ramp over 14 cycles |
|---|---|---|---|
| reference | 3690.5 ms | 8.7 Hz | +7.9% |
| + ring_kv | **2556.2 ms** | **12.5 Hz** | **-0.3%** |

**1.44x.** The ramp collapsing from +7.9% to -0.3% is the more important number: the gather scaled
with pool occupancy, so latency used to grow through an episode and now does not. That is what
makes long-horizon tasks predictable, and it is independent evidence the mechanism is the intended
one.

**Cumulative: 8881 -> 2556 ms, 3.47x, 3.6 -> 12.5 Hz, every step bit-exact.**


---

# 8. RESTATED under the fixed benchmark protocol (2026-08-01)

Every latency number in sections 3, 4, 6 and 7 was measured with **one probe run per arm**. That
is not a valid protocol on this box, and the numbers above are superseded by this section.

## What was wrong

The first probe run after a server starts is up to **37% slower** than steady state -- cuBLAS/cuDNN
algorithm selection and allocator warm-up. `probe_latency` discarded cycle 0 but not the first
*run*, so a single run silently mixes warm-up into the mean. The same configuration measured
2556 ms and 3503 ms on consecutive days with identical flags.

Two things it was NOT, both checked before concluding:

| hypothesis | test | result |
|---|---|---|
| the harnesses disagree | in-process vs websocket, same config | 3517 vs 3503 ms, **0.4%** |
| the GPUs differ | A/A: identical config on GPU 0 and GPU 1 | 3532.6 vs 3510.6 ms, **0.6%**, same clocks |

A separate bug compounded it in the in-process profiler: `drive()` restarted its cycle counter on
every call, so the "measured" cycle ran the cycle-0 workload (4 keyframes, cold VAE encode) and
reported 4893 ms for a 3500 ms cycle.

## The protocol, now the only accepted one

`probe_latency.py --repeats N` (default 3). The first run is **discarded**; the rest are reported
with their spread, and a spread above 5% prints a refusal to quote the number. Any latency claim
must come from this path.

## Restated numbers

Four cumulative configs, one per GPU, probed sequentially, 10 cycles x 3 runs each.

| config | cycle | spread | vs stock | control rate | step |
|---|---|---|---|---|---|
| stock | 8431.5 ms | 0.0% | 1.00x | 3.8 Hz | |
| + substrate (fsdp, empty_cache, debug dumps) | 3994.0 ms | 0.4% | **2.11x** | 8.0 Hz | 2.11x |
| + conditioning prefill | 3567.5 ms | 0.1% | 2.36x | 9.0 Hz | 1.12x |
| + ring KV addressing | **2553.9 ms** | 0.7% | **3.30x** | **12.5 Hz** | 1.40x |

## What changed versus what was published

| claim | published | restated |
|---|---|---|
| stock cycle | 8881 ms | 8431.5 ms |
| substrate passes | 1.92x | **2.11x** |
| conditioning prefill | 1.05x | **1.12x** |
| ring KV | 1.44x | **1.40x** |
| **cumulative** | **3.47x** | **3.30x** |

Two went up and one went down, which is the signature of noise rather than bias -- the old
single-run numbers were not systematically flattering, they were just unreliable. The headline is
lower: **3.30x, not 3.47x.**

Unaffected: every correctness result. Bit-exactness, the 800/800 allocator parity checks, and the
3/3 identical action streams on `put_bottles_dustbin` are equality tests, immune to timing noise.

---

# 9. copy_ breakdown — measured, and the reason not to pursue it (2026-08-02)

Profiled before choosing cache reuse as the next optimization. **Recommendation: do not pursue.**

```
aten::copy_, one control cycle : 74,547 calls, 183.4 ms GPU, mean 2.46 us/call
share of the 2832 ms episode-mode cycle : 6.5%
```

| shape | n | ms | ideal (HBM) | overhead | inferred source |
|---|---|---|---|---|---|
| (2, 32, 3072) | 17264 | 41.6 | 4.05 | **90%** | action hidden state |
| (2, 240, 3072) | 8632 | 39.5 | 15.20 | 62% | video hidden state |
| (2, 240, 24, 128) | 4680 | 28.1 | 8.24 | 71% | video q/k/v, head-split |
| (2, 32, 24, 128) | 9360 | 27.1 | 2.20 | **92%** | action q/k/v, head-split |
| (1, 8, 10) | 11712 | 13.7 | 0.00 | 100% | unexplained, tiny |
| 5 more tiny families | 19936 | 26.9 | ~0.1 | ~100% | unexplained, tiny |

**80% of copy time (147.1 ms) is per-kernel overhead, not bandwidth.** Only 36.3 ms is what moving
the bytes costs. The lever is COUNT, not bytes — the opposite of how this was framed when it went
on the ranking as "46,752 copies, 79 ms GPU".

**Ceiling.** Every copy vanishing is 183.4 ms of 2832 ms = **1.069x**, unreachable by construction.
The realistic target is the four large-tensor families: 136.3 ms, **4.8% of the cycle**, and
capturing even half of it means fusing across rounding points that the block trace lists as
semantic. The tiny tail is 47.1 ms (1.7%) spread over ~29k calls of 80–1280 elements.

**Limits of this attribution, stated so it is not over-read.** `TorchDispatchMode` observed only
4,817 of the 74,547 copies — the remainder are issued below the Python dispatch key — so the
source column is inferred from shapes, not proven from call sites. The tiny families are genuinely
unexplained: 11,712 copies of `[1, 8, 10]` per cycle is ~148 per forward and no block op has that
shape. Not chased further, because 0.5% of a cycle cannot justify it.

A first pass apportioned GPU time by bytes and was wrong: 5.92 GB over 183.3 ms is 32 GB/s, ~100x
under HBM, which is precisely the signal that these copies are overhead-bound rather than
bandwidth-bound.
