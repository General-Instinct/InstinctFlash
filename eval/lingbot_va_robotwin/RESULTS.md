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

Pinned accepted-seed lists are in `/home/ubuntu/iwm_seeds/baseline50` and per-chunk action
streams in `/home/ubuntu/iwm_actions/baseline50`, so a later arm can replay the identical scene
set and be compared per-episode.

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

## 5. What the numbers say about where to go next

After the free 1.92×, a forward still costs 56 ms against a ~3 ms roofline. The remaining
gap decomposes into two very different problems:

1. **Step count.** 77 forwards per chunk is the dominant term and no amount of kernel work
   touches it. Few-step distillation is the only lever with an order of magnitude in it.
   The 51-step *action* loop is the larger half and conditions on a KV cache the 26-step
   video loop already wrote, so the two are not equally compressible.
2. **Per-forward overhead.** ~53 of the remaining 56 ms is not arithmetic. CUDA graph
   capture, `torch.compile`, and an attention kernel that handles a growing KV window are
   the candidates — but note CUDA graphs were previously measured **not** bit-exact on a
   related model, so that one buys speed at the cost of an expensive accuracy certificate.

Two structural observations worth more than either:

- **CFG doubles every forward.** `guidance_scale=5 > 1` duplicates the batch on all 77
  forwards. A guidance-distilled model halves the compute with no scheduling change.
- **The paper's "asynchronous execution" is not in the server.** There are no CUDA streams
  anywhere in the repo; `wan_va_server.py:489-522` runs the video loop, then `:524-561`
  runs the action loop, strictly sequentially.

Finally, a protocol caveat that governs all of the above: **RoboTwin does not score
latency.** `take_action` blocks until the policy returns, so a 77-forward model ties a
4-forward model. Every number in §4 is real, and none of it shows up in §2. Making latency
count needs its own protocol.

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
