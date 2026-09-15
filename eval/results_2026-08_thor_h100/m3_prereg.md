# M3 pre-registration: LingBot-VLA-V2-6B Thor engine, paired closed-loop non-inferiority on RoboTwin 2.0

Declared: 2026-08-26 (UTC), BEFORE any policy episode of this campaign was run and before any
outcome was read. Author: the M3 session (guanming@general-instinct.com's box). This document
freezes the design, the data-handling rules, and the single analysis. Nothing below changes
after data collection starts.

## 1. Question

Is the FULL FlashRT engine for LingBot-VLA-V2-6B on Thor — the M2 ship config — non-inferior
in closed-loop task success to the stock V2 pipeline on the same Thor hardware, on RoboTwin
2.0 demo_clean, at the pre-registered margin of **-0.05 absolute task success**, under the
repo's pre-registered paired analysis
(`/home/ubuntu/InstinctWM/eval/lingbot_va_robotwin/certify_operating_point.py`, unmodified)?

This is the pre-declared decider left open by M2 (`m2_gate.json`, `m2_backbone_gate.json`):
engine actions sit 1-2 bf16-ulp above the per-frame strict max-abs envelope (0.059-0.074 vs
0.047), with p99 inside it; M2's adjudication text says verbatim that "closed-loop
non-inferiority (M3, paired RoboTwin) is the decisive gate for shipping claims". A
NON-INFERIOR verdict here is the shipping gate; NOT ESTABLISHED means the engine does not get
the closed-loop claim, and the margin is not to be relaxed.

Also disclosed up front: the engine's fp8 static activation scales (tier-2 calibration,
`thor:~/thor_t2v2/calib/scales.pt` + `repack_v2_out`) were calibrated on frames from the
pilot-10 task set only. This campaign evaluates all 50 tasks with those scales FROZEN — the
generalization of the calibration is part of what is being tested. No recalibration at any
point.

## 2. Arms (both served from Thor, same websocket protocol, one server process)

* **Control (arm A, "stock")**: the stock V2 pipeline exactly as every Thor V2 number was
  measured (`thor_column/vla2.json` arm_notes): Thor clone `~/lingbot-vla-v2-repo` (sdpa
  loader patch; pure-torch dense-MoE equivalent of the fused kernel — Triton cannot codegen
  for sm_110a), `venv_vla2` (torch 2.11.0+cu130, transformers 4.57.3),
  `LingbotVLAv2Server(MODEL_PATH, use_length=50, chunk_ret=True, use_bf16=True,
  use_fp32=False, use_compile=False)`, `reset('robotwin')` per episode. Checkpoint
  `robbyant/lingbot-vla-v2-6b-robotwin` snapshot `04518557` `global_step_50000/hf_ckpt` —
  the same weights in both arms.
* **Treat (arm B, "engine")**: the FULL FlashRT engine ship config from
  `m2_backbone_gate.json`: engine ViT fp16 (`fmha_strided_full`) + engine LM prefill fp16
  GEMMs (`fwd_bf16_causal` attention) + captured fp8 W+A denoise loop (batched v0.5 MoE with
  shared per-(layer,matrix) scales incl. L13, fp16-source fp32-math router via
  `ffn_xn_fp16`), stock_bf16 step tables + inv_freq rope (M1 fixes), stock-dumped vision rope
  tables (`~/thor_t2v2/p1_dumps/vis_tables.pt`), frozen tier-2 calib static scales
  (`~/thor_t2v2/calib/scales.pt`: `lm_act_scales`, `exp_act_scales`, `moe_down_act_amax`),
  artifacts `~/thor_t2v2/repack_v2_out` (R0-gated repack of the same checkpoint), fp16 LM
  stack from the same checkpoint. Prefill and loop run as CAPTURED CUDA graphs (the
  bit-exact-verified 210.3 ms configuration, `phase_l2` two-graph pattern); graphs are
  (re)captured per prompt change, never inside an episode's step loop.

Both arms run in ONE server process on Thor (`m3_thor_server.py`, the M2 `phase_l2`
coexistence pattern), so preprocessing is BYTE-IDENTICAL by construction: both arms consume
`LingbotVLAv2Server._prepare_model_input(obs)` / `feature_transform.apply`, and both arms'
actions return through the same `feature_transform.unapply`. The wire protocol is the
upstream V2 deploy protocol (msgpack-numpy, `WebsocketClientPolicy`), extended with
`{reset, robo_name, arm, episode_seed, episode_id}` on reset. Simulation clients run on the
H100 box (`/home/ubuntu/RoboTwin`, its `.venv`), GPU 7 only.

## 3. Protocol (from the upstream V2 eval config, cited)

Everything below is taken from the V2 repo's own RoboTwin eval graft
(`experiment/robotwin/start_robotwin_infer_and_eval.sh` +
`experiment/robotwin/eval_policy_client_lingbotvla.py`), not invented here:

* **Task set**: the launcher's full 50-task `task_list_all`, in its order. 50 tasks.
* **task_config**: `demo_clean` (launcher default).
* **Seed rule**: client `--seed 0`, `st_seed = 100000 * (1 + seed) = 100000` (the upstream
  client's own rule, identical to this box's `script/eval_policy.py` convention and its
  >=100000 train/eval disjointness gate).
* **Instruction type**: `unseen` — the upstream client reads `instruction_type` from
  `policy/ACT/deploy_policy.yml` (the config skeleton it is launched with) and the launcher
  never overrides it; that file says `instruction_type: unseen`.
* **Serving config**: `use_length=50`, `chunk_ret=True`, bf16 — the launcher defaults; the
  client executes the full 50-action chunk per infer.
* **Episodes per task**: upstream runs 100; we pre-declare **10 per task** -> **n = 500
  pairs** (the pre-declared pair count for this certificate; T3 precedent). Episode indices
  0..9 in the pinned scene list (Section 4), identical in both arms.
* Video recording OFF (the upstream launcher's effective default; saves wall clock).
* Success = `TASK_ENV.eval_success` within the task's `step_lim`
  (`task_config/_eval_step_limit.yml`), exactly as upstream.

## 4. Scene set (pinned; frozen per task before any policy episode of that task)

Upstream's expert gate (`plan_success and check_success()` on a `play_once()` demo) chooses
scenes run-dependently — the measured cuRobo nondeterminism documented in
`script/eval_policy.py` and the VA 2V/4A prereg §2 defect. M3 closes this with the
accepted-seed cache mechanism already in this repo:

* Cache dir: `/home/ubuntu/iwm_seeds/m3_vla2/<task>_demo_clean_seed0.json`.
* For the 10 tasks whose caches were already built and published in July
  (`InstinctWM_backup_20260731_223220Z/seeds/*_demo_clean_seed0.json`: blocks_ranking_rgb,
  blocks_ranking_size, handover_mic, move_can_pot, move_stapler_pad, open_microwave,
  place_can_basket, place_dual_shoes, place_fan, stack_blocks_three; st_seed=100000,
  n_target=100, `random_seed_patch_active: true`, RoboTwin git c3ddfa8b — the SAME checkout
  still at `/home/ubuntu/RoboTwin`): those caches are copied verbatim and reused.
* For the other 40 tasks: built by the same published tool
  (`InstinctWM_backup_20260731_223220Z/tools/robotwin_seed_cache.py build --n-target 12`),
  i.e. the first 12 expert-accepted seeds counting upward from st_seed=100000. Mechanical,
  no discretion. Each task's cache is built (or copied) BEFORE that task's first policy
  episode; cache building is policy-independent (the expert planner never sees either arm).
* **Certificate scenes = pinned-list indices 0..9 per task** (the first 10 accepted seeds).
* **Keep-on-disagreement**: as in `script/eval_policy.py`, if this run's expert gate
  disagrees with a cached seed, the scene is KEPT and the disagreement recorded — the scene
  set never varies between arms.
* Episode identity: `episode_id := "<task>/<index in the pinned list>"` (0..9), identical
  across arms by construction. Pairing is by `episode_id`, the tool's definition.
* Contamination gate: every scene seed >= 100000; training seeds are all < 1000 (the
  audited manifest bound). Asserted arithmetically by the client; a violation refuses to
  run.

## 5. Noise and instruction pinning (the T3 class of trap, closed)

**Denoise noise.** Stock V2's serving path draws its flow noise from the process-global
torch RNG (`modeling_lingbot_vla_v2.sample_actions: noise = torch.randn(...)`, process
seeded 42 at import) — the noise a given episode gets depends on everything served earlier
in that process lifetime. That is exactly the class the T3 harness closed with per-episode
reseeding. M3's rule, applied identically to BOTH arms by the server:

* Per episode, at reset: `rng = np.random.Generator(np.random.PCG64(episode_seed))` where
  `episode_seed` = the episode's scene seed.
* Per infer call k of that episode: `noise_f32 = rng.standard_normal((50, 55),
  dtype=np.float32)`; `noise_bf16 = torch.from_numpy(noise_f32).to(torch.bfloat16)`.
* Arm A receives it explicitly: `model.sample_actions(..., noise=noise_bf16[None].cuda())`
  (the model's own explicit-noise parameter; nothing else in the stock path is touched).
* Arm B stages the SAME values: `fr._x_t.copy_(noise_bf16.to(fp16))` (bf16 -> fp16 is exact;
  the engine substrate is fp16, exactly as gated in M2).
* The per-call sha256 of the fp32 noise bytes is returned on the wire; the client records
  the full per-call sha list (`noise_shas`) in `episodes.jsonl`. Gate G2 (Section 8) asserts
  PREFIX equality across arms per episode (the arms legitimately make different numbers of
  infer calls when one succeeds earlier; call k's noise is a pure function of
  (episode_seed, k), so the first min(n_A, n_B) shas must match exactly).

Consequence: the noise per (episode, call) is IDENTICAL across arms, standard normal in
distribution (the estimand — stock V2's closed-loop success under its own protocol — is
unchanged in distribution; only the draw is pinned), and the serving path draws NO unseeded
noise. The `--repeat` arm therefore functions as a determinism audit rather than a noise
floor (Section 7); it is kept anyway, per the tool's design, and remains informational.

**Instructions.** The upstream client samples `np.random.choice(descriptions[unseen])` from
the process-global numpy RNG. M3's client reseeds `np.random.seed(episode_seed % 2**32)`
immediately before instruction generation+choice, making the instruction a pure function of
(task, scene seed) — identical across arms. Gate G3 asserts pairwise instruction equality
from the logged strings before the analysis is run.

## 6. Execution plan (serialized; block-alternated; crash-tolerant)

Per task, in the declared 50-task order:

1. Seed cache built/copied if missing (Section 4).
2. **Arm A block**: 10 episodes (indices 0..9), one client process, GPU 7.
3. **Arm B block**: same 10 episodes, one client process, GPU 7.

Alternating arms in task-sized blocks keeps the pairing honest under any slow drift (thermal,
sim-state, Thor clocks). On Thor, the server holds `flock /tmp/thor_gpu.lock` for the
duration of each client connection (= one arm-task block), releasing between blocks. One
Thor server lifetime is intended for the whole campaign; because every episode's noise is a
pure function of its seed (no cross-episode RNG state), a server restart cannot change any
episode's inputs — restarts are permitted and logged, unlike T3 where the global-RNG design
forbade them.

* Results: `/home/ubuntu/iwm_results/m3_vla2_stock/episodes.jsonl` and
  `/home/ubuntu/iwm_results/m3_vla2_engine/episodes.jsonl`, appended one line per episode as
  it completes (a crash loses nothing).
* Resume rule: a re-run of a block skips episode_ids already present in that arm's jsonl
  (`--skip-done`); completed episode records are never regenerated or edited. Harness
  crashes are fixed and the affected block re-run for its missing episodes only; harness
  repair never touches analysis or protocol. All such events go in `/tmp/m3_progress.md`.
* Progress monitoring is by episode COUNT only; no success aggregate is computed or read
  until both arms are complete (Section 9).

## 7. Repeat arm (declared)

After both arms complete: **arm A re-run on the first 10 tasks of the declared order x their
10 certificate episodes** (100 pairs) -> `/home/ubuntu/iwm_results/m3_vla2_stock_repeat/`.
Passed to the tool as `--repeat`. Under this design (deterministic Thor stock — M2's
measured 16-draw null was exactly 0.0 — pinned noise, pinned scenes/instructions,
`random.seed` sim patch active) its expected discordance is ~0; any nonzero discordance
measures residual sim/harness nondeterminism. Per the tool, the floor is informational and
cannot overturn the interval verdict.

## 8. Integrity gates (none can be tuned on outcomes)

* **G1 prompt parity** (per episode, in-server, hard-refuse): the engine's tokenization of
  the instruction (chat template, max_length 72) must equal the stock
  `_prepare_model_input` lang_tokens bit-exactly (valid rows) with the standard dense mask.
  This is the M0 gate made a serving invariant; it closes the V2 analog of the VA 22.7h
  silent prompt-mismatch class.
* **G2 noise parity** (post-run, pre-analysis, hard-refuse): per-episode per-call noise sha
  lists agree on their common prefix across arms for all 500 pairs (see Section 5).
* **G3 instruction parity** (post-run, pre-analysis, hard-refuse): per-episode instruction
  strings equal across arms.
* **G4 capture bit-exactness** (smoke only): captured-graph engine actions equal the eager
  engine actions on identical staged inputs, re-confirming the M2 property in this serving
  path. Also in smoke only: a live stock-vs-engine action maxdelta on real closed-loop
  frames is recorded (expected in the M2 band ~0.06-0.08); it is a report, not a gate.
* **G5 action sanity** (always): non-finite actions are a hard server error. Normalized
  engine actions with absmax > 10 are logged as anomalies (recorded, non-gating — stock
  absmax on real frames is ~4-6).
* **G6 seed floor** (client, hard-refuse): all scene seeds >= 100000.

A G2/G3 failure means the pairing itself is broken: the affected episodes are re-run in both
arms per the Section 6 resume rule (their inputs are pure functions of the seed, so this is
repair, not resampling), and the event is disclosed.

## 9. No peeking

No success rates, deltas, or per-arm aggregates are computed or read for any certificate
episode until BOTH arms are complete and the parity gates pass. Progress is tracked by
episode counts in `/tmp/m3_progress.md`. The analysis in Section 10 runs exactly once.

Disclosed exception — the 2x2 smoke: task `move_can_pot`, pinned-list indices 10 and 11
(scene-disjoint from every certificate episode by construction), 2 episodes per arm, run
first to prove the plumbing (websocket path, pairing, G1/G4, jsonl format). Its outcomes are
read (that is its purpose) and are NOT evidence: smoke records go to
`/home/ubuntu/iwm_results/m3_vla2_smoke_*` and never touch the certificate dirs.

## 10. The single pre-registered analysis

Command, frozen verbatim (the tool is the pre-registered analysis and is NOT modified):

```
python3 /home/ubuntu/InstinctWM/eval/lingbot_va_robotwin/certify_operating_point.py \
  --control m3_vla2_stock --treat m3_vla2_engine \
  --repeat m3_vla2_stock_repeat \
  --margin -0.05 \
  --label "M3: Thor FULL engine (fp16 prefill + captured fp8 loop) vs stock V2, RoboTwin 50x10 paired" \
  --out /home/ubuntu/iwm_distill/thor_t2v2/m3_certificate.json
```

Full stdout is saved verbatim to `/home/ubuntu/iwm_distill/thor_t2v2/m3_certificate.txt`.

* **Margin: -0.05 absolute, NOT to be relaxed under any outcome** (the margin every
  certificate in this repo has used).
* **Verdict rule: the tool's own** — non-inferior iff the MOST CONSERVATIVE of its three
  lower bounds (McNemar-SE normal, iid-episode bootstrap, task-cluster bootstrap) is
  strictly greater than -0.05. Reported EXACTLY as the tool prints it: NON-INFERIOR or NOT
  ESTABLISHED (with the tool's own FRAGILE/robust qualifier).
* Unpaired episodes (should not exist under Section 6; possible only after a disclosed
  irrecoverable harness failure) are dropped and reported by the tool, never imputed.

## 11. Power, and what happens on each outcome

With pinned noise/scenes/instructions, discordance comes only from ulp-scale numeric deltas
amplified by closed-loop divergence. At true delta 0 the gate passes for discordance up to
~30% at n=500; at true delta -0.02 it passes only if discordance stays under ~12%. This is a
real chance of failure and is accepted.

* **NON-INFERIOR** -> `m3_certificate.{txt,json}` closes the M2 open question;
  `thor_column/vla2.json` `engine_full_arm` gains the closed-loop verdict (quoted exactly);
  research log appended.
* **NOT ESTABLISHED** -> reported exactly so, margin not relaxed, the engine ships WITHOUT
  the closed-loop non-inferiority claim; the per-task discordance table is published for
  diagnosis; any extension requires a new pre-registration.
* Latency is NOT an outcome of this certificate (M2 owns the latency numbers); `server_ms`
  is recorded per infer for the record only.

## 12. Preflight gates (run before the smoke; results appended below)

1. Thor server up, both arms loaded, one process; metadata banner records ship-config
   details, checkpoint path, artifact paths, calib sha.
2. G1 passes on the smoke episodes.
3. G4 passes on a smoke frame (captured == eager, bit-exact).
4. Seed caches present for the smoke task; certificate caches copied/built per Section 4.
5. Client venv sanity: RoboTwin `.venv` (numpy 1.26.4, websockets 16.1.1), sim on GPU 7
   only; `envs/_base_task.py` `random.seed` patch present (disclosed patch #2).

---

### Preflight results (appended after gates ran, before any certificate episode)

Smoke (2026-08-26, move_can_pot pinned idx 10-11 = seeds 100012/100013, 2 eps x 2 arms,
sim GPU 7, one Thor server lifetime, ws://100.68.159.80:29940):

1. Server up, both arms in one process: PASS (stock load 55.1 s, engine artifacts 15.4 s;
   flock held during load and per connection; banner records model path + arms + noise rule).
2. G1 prompt parity: PASS (engine chat-template ids == stock lang_tokens, dense mask).
3. G4 capture bit-exactness: PASS on all 6 engine infers (captured replay == eager rerun,
   bit-identical actions).
4. Seed caches: move_can_pot copied from the July backup (st_seed=100000, 100 accepted);
   9 further pilot caches copied; 40 to build in-campaign per §4.
5. Client env: PASS (RoboTwin .venv, numpy 1.26.4, websockets 16.1.1; _base_task.py
   random.seed patch present; sim confirmed on GPU 7).
6. Pairing plumbing: PASS — per-episode noise digests IDENTICAL across arms
   (7202fbb1cca9d136 / 07ee01b70e741a7f), per-call sha lists identical, instructions
   identical, G6 min seed 100012; m3_parity_gates.py on the smoke dirs: PASS.
7. Smoke-only live parity report (not a gate): stock-vs-engine normalized-action maxdelta
   per infer = 0.051-0.105 on these closed-loop frames (M2's pilot-frame band was
   0.056-0.078; first-chunk frames read 0.092/0.105, i.e. ~2 more bf16-ulp on unseen
   frames). This is precisely the question the certificate adjudicates; engine norm absmax
   3.7-4.9 (sane), no anomalies.
8. Timing for projection: ~27-29 s/episode steady-state on this 400-step task per arm
   (episode wall includes the expert demo). Engine infer_ms in smoke (1.31 s) is inflated
   by the smoke-only G4 eager rerun + stock cross-check; cert-mode engine infer is the
   ship-config path (~0.2 s) + preprocessing.

All preflight gates PASS; the certificate campaign starts with no protocol change.
