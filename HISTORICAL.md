# Historical material: negative results vs archived implementations

Two different things were sitting in one pile. They are kept for opposite reasons and should be read
differently, so they are separated here and marked in each module's docstring with a `STATUS:` line
that `tests/test_shipped_config.py` enforces.

| | **negative result** | **archived implementation** |
|:--|:--|:--|
| what it is | a hypothesis that was tested and **refuted** | working, correct code that simply is not enabled |
| why it is kept | so it is not re-proposed | so it can be picked up if the conditions change |
| what to do with it | read the refutation before spending a day on the same idea | check whether its conditions now hold |
| the danger if deleted | someone re-derives it and re-learns the same thing | the work is redone from scratch |
| the danger if promoted | a known-bad idea gets shipped | a correct pass ships without a current gate |

Neither kind runs. Every module below defaults off, and none appears in
[`shipped_configuration()`](instinctwm/verify/released.py).

---

## Negative results

Tested, refuted, kept as the refutation.

### `passes/lingbot/cfg_elision.py`
`guidance_scale=5` and `action_guidance_scale=1`, so the action stream's CFG output *is* discarded by
`action_noise_pred[:1]` — which makes `dead_outputs` a true statement about **output usage** and not
about dead compute. A two-axis liveness test found branch 1 **live on both**: corrupting its returned
value moved actions by **5.64**, suppressing only its writes to the shared KV pool moved them by
**5.39**, against a chunk-to-chunk movement of **1.03**. Both branches write the pool and the video
stream at scale 5 reads branch 1. This was the largest duplicated-execution candidate in the system —
~98 ms of device work and half the host dispatches.

### `passes/lingbot/persistent_graph.py`
Every correctness gate passes: byte-identical KV writes, `max |Δ action| = 0` over 45 cycles spanning
saturation, reset isolation, captures 270 → 238. The latency gate refuses it — **503.5 ms against
351.4 ms with capture off, 1.43× slower**. The plan buffer recovers 432 of the 585 ms capture penalty
and it does not matter: 5.3 surviving captures at ~111 ms each exceed the whole cycle. Full record in
[LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md), including the *second,
independent* operational reason — graph eviction does not return its private pool, so a 50-task run
climbs to the 80 GB ceiling and OOMs.

### `passes/lingbot/fused_qkv.py`
1.9% predicted, **0.2% slower** measured. Its own fail-closed per-shape certificate disproved the
invariant it was built on: **M=7 differs in 55 of 64,512 words**, so cuBLAS `tile_k` invariance is not
universal across the served envelope. The certificate machinery is reusable and the pass is not.

### `passes/lingbot/forward_scratch.py`
Struck on **Cosmos3-Edge**, the model it was written for. Removing all 896 allocations and the whole
0.97 GiB buys **1.010× on eager and 1.000× on the shipped (graph) path** — capture already bakes the
allocations into its private pool, so it is not merely small, it is *subsumed*. The manifest's byte
estimate was also wrong by 2.1× (2.08 GB claimed, 0.97 GiB true) because `get_all_seq` runs on K and V,
which GQA narrows to 8 heads.

### `passes/lingbot/static_partition_hoist.py`
**Obsolete — upstream implemented it.** `init_sequence_pack` became `SequencePackMetadata` +
`prepare_sequence_pack_metadata` + a `prepared_metadata=` parameter, and `cosmos3_vfm_network.py:1017`
passes it on the served path. What survives of the original finding is only the `.tolist()`
device→host sync at `runtime.py:243` and the assert-only product it feeds.

### `backends/rope.py` — the fused RoPE Triton kernel
Bit-exact after fixing a double-rounding bug (torch narrows fp64→bf16 *via* fp32), **1.10× at region
scale**, and the region was **0.3% of the cycle**. The first of the region-vs-cycle failures that
eventually produced the regime model. The kernel-region framework around it is infrastructure and
stays; the kernel is the negative result.

### Measurement-only probes carrying refutations
`probe_action_terminal.py` and `probe_action_terminal_liveness.py` — the terminal action forward is
dead for ~38 cycles and **live thereafter**, because the ring wrap evicts a slot and advances `start`
in a way `clear_pred_cache`'s count rollback does not undo. Annotated in place; a run that stops before
the wrap reports the opposite.

---

## Archived implementations

Correct, working, not enabled. Nothing is wrong with these.

### `passes/lingbot/step_scope_cast.py`
BITEXACT. Restores `temb.float()` to per-forward scope, removing **1,740 casts per cycle**. Not enabled
because the cycle effect is **0.66%** and one ABBA arm failed convergence at 6.4%. It would need a
current gate before shipping, and the return does not justify one. Kept because if a future operating
point makes per-forward cast scope matter, the implementation already exists and is verified exact.

### `runtime/state/scratch.py` — the scope-bumped arena
Infrastructure rather than a pass, and the reason it is listed here: the pass that used it
(`forward_scratch`) is a negative result, so the arena has no current consumer. The mechanism is sound
and general.

---

## Passes that are released but not recommended

These are neither — they are in the ledger, and the ledger is not rewritten. See
`DISPOSITIONS` in [`verify/released.py`](instinctwm/verify/released.py):

- **P005 `graph_block_stack`** — released BITEXACT at 1.380× on a ~2.5 s cycle. `NOT_RECOMMENDED`
  today: capture measures 1.43× slower at the Fast operating point.
- **P006 `stable_state_pools`** — released at 1.520×. `NOT_RECOMMENDED` only because it exists to let
  P005's graphs survive a reset, and P005 is not recommended. Not refuted on its own; it simply has
  nothing to do.

The distinction matters: **released ≠ recommended**, and conflating them is what let the repository
assert a shipped configuration it was not running.

## Where the corresponding write-ups live

| document | what it records |
|:--|:--|
| [LAYER5_COMPLETE.md](LAYER5_COMPLETE.md) | Layer 5 screened and closed; zero fallback kernels remain |
| [LAYER5_GRAPH_PERSISTENCE_RESULT.md](LAYER5_GRAPH_PERSISTENCE_RESULT.md) | graph persistence, frozen, both reasons |
| [LAYER6.md](LAYER6.md) | host-dispatch inventory, and the refutation of its own ranking term |
| [LAYER6_GAPS.md](LAYER6_GAPS.md) | the 139 ms of device idle, diffuse, the eager floor |
| [LAYER6_REGIMES.md](LAYER6_REGIMES.md) | the two regimes and the exchange rates that price everything above |
| [LAYER5_QKV_EXACTNESS.md](LAYER5_QKV_EXACTNESS.md) | why fused QKV's bit-exactness was an accident |
| [LAYER5_CAST_FAMILY.md](LAYER5_CAST_FAMILY.md) | why the cast family does not generalize |
| [INVENTORY.md](INVENTORY.md) | ships / infrastructure / historical, for the whole tree |
