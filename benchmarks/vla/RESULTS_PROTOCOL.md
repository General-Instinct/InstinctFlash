# Published latency protocols

These notes describe the historical August 2026 README measurements.
The later simulator experiments and native Thor probes have separate protocols.

Reproduce H100 pairs with `examples/<family>/reproduce_h100.sh`.

H100 cells are the 2026-08-24 sweep; LingBot-VLA-V2 is the 2026-08-28 re-sweep on a different
H100-80GB box (4xH100 host), both cells remeasured there, after its default arm gained the
vision/prefill graphs and GPU preprocessing. The re-swept VLA-4B, GR00T and pi05 pairs
reproduced their rows' class on that box and keep their published cells.

‡ LingBot-VA cycle latency has two regimes within an episode: early (the ring-KV pool still filling,
cycles 1–36; a typical RoboTwin episode ends before the pool saturates) and saturated (pool full,
cycles ≥ 37). The LingBot-VA cells above are the early regime (cycles 2–8 on Thor, 2–12 on H100).
Measured to saturation (48-cycle episodes × 3 runs, run 0 discarded, real message order), the Thor
cells read 31533 → 8250 ms, 3.82× and 31533 → 1382 ms, 22.8×; on the 4xH100 host the same-computation
ratio is 4.10× and 2V/4A is 27.7×. Every ratio holds or rises at saturation — the vendor server's cycle
grows more with pool size than ours — while the Thor absolute ms above understate a saturated episode
by 1.5–1.8×; on H100 our default-schedule chain is regime-flat and the vendor server is +20 %.

The historical LingBot-VLA-V2 H100 capture row now reproduces with an explicit
`tier_ceiling="numeric", placement="in_process"`: its numerical-envelope self-check
is no longer labeled BITEXACT by the planner. The measured cells remain historical.
