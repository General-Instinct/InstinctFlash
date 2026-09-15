# M3-EXT pre-registration: evidence extension of the V2 Thor engine closed-loop certificate to pooled n≈1100

Declared: 2026-08-28 (UTC), BEFORE any extension policy episode was run and before any
extension outcome existed. Author: the M3-EXT session (guanming@general-instinct.com's box).
This document freezes the design, the data-handling rules, and the single analysis. Nothing
below changes after data collection starts. Precedent mirrored deliberately: the VA 2V/4A
certificate extension (`iwm_distill/thor_column/va_2v4a_cert_extension_prereg.md`) — interim
disclosure, mechanical scene pinning, pooled single analysis, honest power/flip-risk note.

## 1. Question

Same question as M3 (`m3_prereg.md` §1), with more evidence: is the FULL FlashRT engine for
LingBot-VLA-V2-6B on Thor (the M2 ship config) non-inferior in closed-loop task success to
the stock V2 pipeline on the same Thor hardware, on RoboTwin 2.0 demo_clean, at the
pre-registered margin of **-0.05 absolute task success**, under the repo's pre-registered
paired analysis (`/home/ubuntu/InstinctWM/eval/lingbot_va_robotwin/certify_operating_point.py`,
unmodified)?

Why extend: the n=500 verdict passed but is FRAGILE (slack +0.0040 < 0.005) — one adverse
discordant pair sits between the verdict and its reversal. A shipping claim should not rest
on that. This extension adds pre-registered evidence; it does not re-litigate the margin,
the tool, or the protocol.

## 2. The disclosed interim look (n=500) — reported, and superseded by the pooled analysis

`m3_certificate.txt` (2026-08-26, the M3 pre-registered analysis, run once, verbatim):

```
  control   m3_vla2_stock            0.9060
  treat     m3_vla2_engine           0.8860
  paired episodes 500   tasks 50
  delta -0.0200   discordant 15 for / 25 against (8.0%)
  exact McNemar two-sided p      0.1539   (no detectable difference)
  McNemar-SE normal 95% CI       [-0.0448, +0.0048]
  bootstrap 95% CI, iid episodes [-0.0440, +0.0040]
  bootstrap 95% CI, TASK cluster [-0.0460, +0.0040]  <-- primary (most conservative)
  one-sided normal p vs margin   0.00885
  NOISE FLOOR from two runs of the CONTROL configuration (100 pairs):
    discordance 0+0 = 0.0%, delta +0.0000
  margin declared -0.050   primary lower bound -0.0460 (bootstrap_task_cluster)   slack +0.0040
  VERDICT: NON-INFERIOR  (FRAGILE: under 0.005 of slack)
```

This look is disclosed here and is **superseded by the single pooled analysis in §10 as the
headline aggregate**. The n=500 certificate files are kept unmodified as history
(`m3_certificate.{txt,json}`, and the n=500 fields in `thor_column/vla2.json`). Declared up
front, both directions: if the pooled verdict is NOT ESTABLISHED, that supersedes the n=500
pass and the engine loses the closed-loop claim — the fragile n=500 pass is NOT retained as
a fallback headline; if the pooled verdict is NON-INFERIOR (robust or fragile), it is
reported exactly as the tool prints it. Never spin, either way.

## 3. Design of the extension

* **+600 new pairs: 12 new episodes per task x 50 tasks, both arms**, pooled with the
  existing 500 for a single primary analysis at n≈1100 (exact n = paired episodes the tool
  reports after drops; target 1100).
* **Arms: byte-identical in configuration to M3** (`m3_prereg.md` §2, unchanged and not
  restated in full): same Thor box (100.68.159.80), same one-process dual-arm server
  `m3_thor_server.py` (sha256 recorded at preflight and compared to the repo copy
  `iwm_distill/thor_t2v2/m3/m3_thor_server.py`), same checkpoint
  (`robbyant/lingbot-vla-v2-6b-robotwin` snapshot `04518557` `global_step_50000/hf_ckpt`),
  same frozen tier-2 calib scales (NO recalibration at any point), same repack artifacts,
  same captured-graph ship config, same websocket protocol, same per-episode noise and
  instruction pinning rules (§5 of m3_prereg, applied identically to both arms). Server
  restarts remain permitted and logged (noise is a pure function of episode seed; no
  cross-episode RNG state).
* **Margin -0.05, tool unmodified, verdict rule = the tool's own** (most conservative of
  its three lower bounds, strictly greater than the margin).

### 3b. Execution-substrate change (disclosed): sim host moves to the 4xh100 box

At declaration time the local 8xH100 box is fully occupied by a sibling campaign
(8-GPU `wan_va.train` fans training, 13.5h elapsed, plus its checkpoint eval on GPU 7 — the
GPU the M3 sim used). Per the standing compute plan, the sim harness for this extension runs
on the **4xh100 box (tailnet `ubuntu@4xh100`, 100.122.78.46)**:

* **GPUs 2 and 3 only** — GPUs 0-1 belong to another campaign and are not touched
  (`/tmp/gpu01.lock` on that box is respected; it is checked before any launch and the
  campaign refuses GPUs 0-1 regardless of the lock's presence).
* RoboTwin is stood up at **`/home/ubuntu/RoboTwin_m3`** on the 4xh100 box as an exact
  rsync of the local `/home/ubuntu/RoboTwin` working tree — same git commit `c3ddfa8b`,
  same two disclosed patches (`envs/_base_task.py` random.seed patch #2;
  `script/eval_policy.py` mods), same M3 client
  (`script/eval_policy_client_m3_vla2.py`, unmodified — sha256 must equal the local copy),
  same `.venv` (python3.10 substrate provided via a uv-managed CPython 3.10; the venv's
  `python` symlink and `pyvenv.cfg home` are repointed — interpreter plumbing only). The
  existing `/home/ubuntu/RoboTwin` on the 4xh100 box belongs to the pi05 campaign and is
  NOT touched. The curobo editable-install `.pth` in the copied venv is repointed from
  `/home/ubuntu/RoboTwin/envs/curobo/src` to `/home/ubuntu/RoboTwin_m3/envs/curobo/src`
  (preflight asserts `curobo.__file__` resolves under `RoboTwin_m3`). The NVIDIA Vulkan ICD
  (absent on that box) is installed system-wide (`/usr/share/vulkan/icd.d/nvidia_icd.json`,
  copied from the local box; same driver 580.105.08 on both).
* Consequence for inference: the paired design is unaffected — within every pair, both arms
  consume byte-identical preprocessing from the same server process on the same Thor, and
  the sim host is identical within pair by construction. What the box change adds is a
  disclosed assumption for POOLING: the engine-vs-stock paired delta is exchangeable across
  sim hosts (same repo, same commit, same seeds discipline, same GPU model and driver; the
  original 500 pairs rendered on the local box, the 600 extension pairs on the 4xh100 box).
  The task-cluster bootstrap in the tool is the pre-registered guard against per-cluster
  heterogeneity; no additional analysis is invented for this.
* The extension's expert-gate acceptance (cache building, §4) also runs on the 4xh100 box.
  Acceptance is policy-independent (the expert planner never sees either arm), so this
  affects which scenes enter the pinned lists, not either arm's chances on them.

## 4. Scene set (pinned; mechanical; disjoint from ALL existing evidence and smoke)

Extension scenes = **pinned-list indices 12..23 per task** in the M3 accepted-seed caches
(`/home/ubuntu/iwm_seeds/m3_vla2/<task>_demo_clean_seed0.json`, st_seed=100000):

* Indices 0..9 are the existing n=500 certificate scenes — excluded.
* Indices 10..11 are smoke-reserved across ALL tasks (move_can_pot 10-11 carried the M3
  smoke, whose outcomes were read) — excluded from evidence everywhere, uniformly.
* Disjointness from every existing certificate pair, and from the smoke, is therefore by
  construction; `episode_id := "<task>/<pinned index>"` (12..23) collides with no existing
  id (existing ids are `<task>/0..9`).
* **The 10 pilot-cache tasks** (100 accepted seeds each, built July, published): indices
  12..23 are already determined and are pinned here, now, before anything runs:

  | task | seeds (indices 12..23) |
  |---|---|
  | blocks_ranking_rgb | 100012-100023 (contiguous) |
  | blocks_ranking_size | 100012, 100013, 100014, 100015, 100016, 100018, 100019, 100020, 100021, 100022, 100023, 100024 |
  | handover_mic | 100020, 100021, 100022, 100026, 100027, 100030, 100031, 100032, 100034, 100035, 100038, 100039 |
  | move_can_pot | 100014-100025 (contiguous) |
  | move_stapler_pad | 100013, 100014, 100015, 100016, 100017, 100019, 100020, 100021, 100022, 100023, 100024, 100025 |
  | open_microwave | 100019, 100020, 100022, 100023, 100028, 100030, 100031, 100032, 100033, 100036, 100037, 100038 |
  | place_can_basket | 100014, 100016, 100017, 100018, 100020, 100022, 100024, 100025, 100026, 100028, 100029, 100030 |
  | place_dual_shoes | 100023, 100025, 100026, 100027, 100028, 100029, 100030, 100031, 100033, 100034, 100037, 100038 |
  | place_fan | 100014, 100015, 100016, 100017, 100019, 100020, 100021, 100023, 100024, 100025, 100026, 100027 |
  | stack_blocks_three | 100014-100025 (contiguous) |

* **The other 40 tasks** hold exactly 12 accepted seeds (indices 0..11), so their caches
  are EXTENDED before use by the declared mechanical rule, no discretion: **the first 12
  additional accepted seeds counting upward from max(attempted seed in the existing
  cache) + 1**, produced by the same expert gate via the process-isolated wrapper already
  in the repo (`iwm_distill/thor_t2v2/m3/m3_build_cache_batched.py`, which calls the July
  builder's `expert_outcomes()` unmodified), run on 4xh100 GPU 3. Extension is
  APPEND-ONLY: entries 0..11 of `accepted_seeds` are byte-preserved (asserted before the
  extended cache replaces the file; the pre-extension file is kept as `*.pre_ext.json`).
  Each task's extended cache lands BEFORE that task's first extension policy episode (the
  M3 §4 discipline: cache building is policy-independent). When the last cache lands, a
  manifest (per-task sha256 + seed lists) is written to
  `/home/ubuntu/iwm_seeds/m3_vla2/_ext_manifest.json` and its sha256 appended to this
  document's preflight appendix.
* **Keep-on-disagreement** unchanged: if this run's expert gate disagrees with a pinned
  seed, the scene is KEPT and the disagreement recorded. An expert demo that never
  completes in 3 attempts skips the episode (dropped-not-imputed at analysis) — the
  client's existing logged behaviour, unchanged.
* Contamination gate G6 unchanged: every scene seed >= 100000, asserted by the client.

## 5. Noise and instruction pinning

Unchanged from m3_prereg §5, verbatim rules, applied by the same unmodified server and
client code: per-episode noise = pure function of the episode's scene seed (PCG64,
per-call standard normal, sha-logged, G2 prefix equality across arms); instruction = pure
function of (task, scene seed) via the np.random reseed before
`generate_episode_descriptions` (G3 equality across arms); `instruction_type: unseen`.
New seeds mean new draws — same estimand, same mechanism.

## 6. Execution plan (serialized; block-alternated; crash-tolerant; NO exit-with-instructions)

Per task, in the same declared 50-task order (`task_list_all`):

1. Extended seed cache present (pilot-10: already; others: built per §4 by the prebuilder
   on GPU 3, which runs ahead in the same task order).
2. **Arm A block (stock)**: 12 episodes (indices 12..23), one client process, 4xh100 GPU 2.
3. **Arm B block (engine)**: same 12 episodes, GPU 2.

Results (on the 4xh100 box, mirrored to the local box continuously):
`/home/ubuntu/iwm_results/m3_vla2_ext_stock/episodes.jsonl` and
`/home/ubuntu/iwm_results/m3_vla2_ext_engine/episodes.jsonl`, one line per episode as it
completes. Resume rule unchanged (episodes.jsonl is the ledger; `--skip-done`; completed
records never regenerated or edited; harness repair never touches analysis or protocol,
and every such event is logged).

**The M3 stall postmortem is binding.** The campaign runner
(`iwm_distill/thor_t2v2/m3/m3_ext_run_campaign.sh`, on the 4xh100 box) scripts the FULL
completion sequence — cache extensions -> all 50 task blocks -> repeat-arm top-up (§7) ->
DONE marker — in an outer loop that never exits while declared work remains. A watchdog on
the 4xh100 box polls the single invariant "uncompleted work exists AND no worker alive"
and relaunches the runner, re-armed unconditionally every wakeup. A second watchdog/syncer
on the local box mirrors results+progress, monitors/relaunches the Thor server if it dies
(restart-tolerant by design, logged), and when the DONE marker and complete ledgers
arrive, runs the §8 gates -> §10 pooling -> §10 frozen analysis, exactly once, scripted.
No stage ends by printing instructions for a human.

Thor serving: same flock discipline (`/tmp/thor_gpu.lock` held during load and per client
connection). Progress is tracked by episode COUNT only in `/tmp/m3_ext_progress.md`
(4xh100, mirrored to the local box same path), per block. The ~12h wall projection is
written to the progress file at launch.

## 7. Repeat arm top-up (declared)

After both extension arms complete: **stock re-run on the first 10 tasks of the declared
order x their 12 extension episodes** (120 pairs) ->
`/home/ubuntu/iwm_results/m3_vla2_ext_stock_repeat/`. Pooled with the existing 100-pair
repeat (`m3_vla2_stock_repeat`) into `m3_vla2_pool_stock_repeat` and passed to the tool as
`--repeat`. This re-measures the determinism floor ON THE NEW SIM HOST (the M3 floor on the
local box was exactly 0.0%). Per the tool, the floor is informational and cannot overturn
the interval verdict.

## 8. Integrity gates (unchanged; none can be tuned on outcomes)

G1 (server-side prompt parity, hard-refuse), G2 (noise-sha prefix parity), G3 (instruction
parity), G4 (capture bit-exactness, smoke only), G5 (action sanity; absmax>10 logged
non-gating), G6 (seed floor >= 100000, hard-refuse) — all exactly as m3_prereg §8, same
unmodified code paths. G2/G3/G6 run post-run pre-analysis via the unmodified
`m3_parity_gates.py` on the POOLED dirs:

```
python3 /home/ubuntu/iwm_distill/thor_t2v2/m3/m3_parity_gates.py \
  --control m3_vla2_pool_stock --treat m3_vla2_pool_engine --n 0
```

`--n 0` because the pair COUNT is not a parity gate: dropped episodes
(expert-demo-never-completes skips, or one-arm absences) are dropped-not-imputed and
reported by the ANALYSIS tool itself (`n_pairs`, `dropped_unpaired` — quoted verbatim in
the certificate); the gate script's "pairs checked / control-only / treat-only" lines are
the disclosure. What gates hard-refuse on: any G2 noise-prefix mismatch, any G3
instruction mismatch, any seed mismatch within a pair, or a G6 seed-floor violation.
A G2/G3 failure means broken pairing: affected episodes re-run in both arms per §6 resume
(repair, not resampling), disclosed.

## 9. No peeking

No success rates, deltas, or per-arm aggregates are computed or read for any extension
episode until both arms are complete, pooling is done, and the parity gates pass. Progress
is episode counts only. The §10 analysis runs exactly once.

Disclosed exception — the new-box smoke: task `move_can_pot`, pinned indices 10-11 (the
same smoke-reserved scenes M3 used; scene-disjoint from all evidence by §4), 2 episodes per
arm on the 4xh100 sim host, run first to prove the moved plumbing (vulkan/render, expert
demo, websocket to Thor, G1/G4, jsonl). Outcomes are read (that is its purpose) and are NOT
evidence: records go to `/home/ubuntu/iwm_results/m3_vla2_smoke2_*` and never touch
certificate dirs.

## 10. Pooling rule and the single pre-registered analysis

Pooling is concatenation, nothing else: `m3_vla2_pool_stock/episodes.jsonl` = the n=500
`m3_vla2_stock/episodes.jsonl` records + the `m3_vla2_ext_stock/episodes.jsonl` records,
verbatim, no record edited, dropped, or imputed (ids are disjoint by §4, so no remap
exists at all — simpler than the VA extension's remap, and declared as such).
`m3_vla2_pool_engine` likewise; `m3_vla2_pool_stock_repeat` = `m3_vla2_stock_repeat` +
`m3_vla2_ext_stock_repeat`. Pairing stays by `episode_id`, the tool's definition.

Command, frozen verbatim (the tool is the pre-registered analysis and is NOT modified):

```
python3 /home/ubuntu/InstinctWM/eval/lingbot_va_robotwin/certify_operating_point.py \
  --control m3_vla2_pool_stock --treat m3_vla2_pool_engine \
  --repeat m3_vla2_pool_stock_repeat \
  --margin -0.05 \
  --label "M3-EXT: Thor FULL engine vs stock V2, RoboTwin paired, pooled n~1100 (pre-registered extension of the FRAGILE n=500)" \
  --out /home/ubuntu/iwm_distill/thor_t2v2/m3_certificate_pooled.json
```

Full stdout saved verbatim to `/home/ubuntu/iwm_distill/thor_t2v2/m3_certificate_pooled.txt`.

* **Margin -0.05 absolute, NOT to be relaxed under any outcome.**
* **Verdict rule: the tool's own**, reported EXACTLY as printed (NON-INFERIOR with its
  FRAGILE/robust qualifier, or NOT ESTABLISHED).
* Unpaired episodes are dropped and reported by the tool, never imputed.

## 11. Power, flip risk, and what happens on each outcome

Under the interim point estimates (true delta -0.020, ~8% discordance), pooling to n=1100
gives SE ~= sqrt(88)/1100 ~= 0.0085 and an expected primary lower bound ~= -0.037 to -0.038
(the cluster bootstrap ran ~0.001-0.002 wider than normal at n=500), i.e. expected slack
~= +0.012 — comfortably past the 0.005 FRAGILE line. Accounting for sampling of the 600 new
pairs (sd of the pooled delta ~= 0.006): **P(NON-INFERIOR) ~= 95%, P(robust slack >= 0.005)
~= 85-90%** if the true delta is the observed -0.020.

The honest flip risk: the n=500 interval is consistent with true deltas down to ~-0.045.
If the true delta is -0.030, P(pooled pass) drops to ~85-90% (robust ~2/3); at -0.040,
P(pooled pass) ~= 35-40%. **Adding evidence can flip the aggregate verdict to NOT
ESTABLISHED. That risk is accepted and pre-declared**: the pooled verdict supersedes the
n=500 verdict in EITHER direction and is reported verbatim, never spun.

* **NON-INFERIOR (robust)** -> the certificate hardens; `thor_column/vla2.json`
  `engine_closed_loop_certificate` gains the pooled fields (n=500 history kept),
  `final_table.md` V2 tier cell updated, eval archive copies updated, research log
  appended (both copies).
* **NON-INFERIOR (still FRAGILE)** -> reported exactly so; the claim keeps the FRAGILE
  qualifier; no further extension without a new pre-registration.
* **NOT ESTABLISHED** -> reported exactly so, margin not relaxed, the engine loses the
  closed-loop non-inferiority claim (the fragile n=500 pass is superseded, not quoted as a
  shield); the per-task discordance table is published for diagnosis.
* Latency is NOT an outcome of this certificate; `server_ms`/`infer_ms` recorded for the
  record only.

## 12. Preflight gates (run before the smoke and before any certificate episode; results appended below)

1. 4xh100 stand-up integrity: `RoboTwin_m3` rsync complete; client sha256 equals the local
   `script/eval_policy_client_m3_vla2.py`; `envs/_base_task.py` random.seed patch present;
   venv functional under the uv CPython 3.10 (numpy 1.26.4, websockets 16.1.1);
   `curobo.__file__` under `RoboTwin_m3`; NVIDIA Vulkan ICD active (sapien offscreen
   render succeeds); sim on GPU 2 only; GPUs 0-1 untouched.
2. Thor server up (relaunch — the M3 server process has since exited; restarts are
   in-protocol and logged), both arms in one process, banner recorded;
   `m3_thor_server.py` sha256 equals the repo copy.
3. G1 passes on the smoke episodes; G4 captured==eager bit-exact on smoke engine infers.
4. Smoke pairing plumbing: per-episode noise digests and instructions identical across
   arms; G6 min seed >= 100000.
5. Seed caches: pilot-10 already >= 24 accepted (pinned in §4); the 40 extensions build
   per §4 with append-only assertion (first verified on the first extended task before its
   blocks run).
6. Wall projection written to `/tmp/m3_ext_progress.md` at launch (~12h: ~600 x 2 x ~28s
   policy episodes ~= 9.5h serialized + cache extensions overlapped on GPU 3 + 120
   repeat episodes ~= 1h + gates/pooling/analysis minutes).

---

### Preflight results (appended after gates ran, before any certificate episode)

All run 2026-08-28 22:0x-22:3xZ, before any extension certificate episode:

1. **4xh100 stand-up: PASS.** rsync of the full local RoboTwin working tree ->
   `4xh100:/home/ubuntu/RoboTwin_m3` (31.1 GB, to-chk=0). Client
   `script/eval_policy_client_m3_vla2.py` sha256
   `5d49334f63545079cafb5810339ff3e7247bbf3cdfd7dac623405532178af89d` == local, byte-identical.
   `envs/_base_task.py` random.seed disclosed patch #2 present. Venv functional under uv
   CPython 3.10.20 (`numpy 1.26.4`, `websockets 16.1.1`, torch 2.4.1+cu121);
   `curobo.__file__` resolves under `RoboTwin_m3` (editable .pth repointed, prebuilt
   cpython-310 CUDA .so carried in-tree). NVIDIA Vulkan: the box had only the headless
   driver (vulkan fell back to llvmpipe, `ErrorExtensionNotPresent`); installed
   `libnvidia-gl-580-server=580.105.08-0lambda0.24.04.1` (exact match to the running
   580.105.08 driver) + the `nvidia_icd.json` ICD -> sapien offscreen render on GPU 2:
   PASS (finite 64x64x4 image). Sim processes confirmed on GPUs 2/3 only; GPUs 0-1 at
   0 MiB throughout.
2. **Thor server: PASS.** Relaunched (the M3 server process had exited since 8-26; restart
   is in-protocol and logged): stock arm loaded 52.0 s, engine artifacts 15.4 s from
   `repack_v2_out`, banner records ship config + checkpoint snapshot 04518557 + the noise
   rule; `m3_thor_server.py` sha256
   `c7269403e3e9a41d9334fc23f60735bc380ed57705cd2c043f04e70a867401e8` == the repo copy.
3. **G1 prompt parity: PASS** on all smoke episodes (`g1_prompt_parity_ok: true`).
   **G4 capture bit-exactness: PASS** on all 6 engine smoke infers (captured == eager).
4. **Smoke pairing plumbing: PASS** (move_can_pot idx 10-11, seeds 100012/100013, 2 eps x
   2 arms -> `m3_vla2_smoke2_*`): seeds, instructions, and full noise-sha lists identical
   across arms; per-episode noise digests `7202fbb1cca9d136` / `07ee01b70e741a7f` —
   BIT-IDENTICAL to the M3 preflight digests on the local box, i.e. the moved client +
   same server reproduce the exact noise stream. G6 min seed 100012. Smoke-only live
   parity report (not a gate): stock-vs-engine normalized-action maxdelta 0.051-0.105 per
   infer — the same band as the M3 smoke (0.051-0.105); engine norm absmax 4.0-4.9, sane.
5. **Cache extension mechanics: PASS.** First extended task (lift_pot) ran the §4 rule on
   4xh100 GPU 3 via `m3_ext_extend_cache.py` (append-only asserted in-code before the file
   is replaced; pre-extension file kept as `.pre_ext.json`); verified below at launch
   boundary. Pilot-10 caches already hold >= 24 accepted seeds (pinned in §4 above).
6. **Timing for projection**: smoke episodes 23.8-51.3 s wall (first episode per arm pays
   asset warm-up; engine smoke pays the smoke-only G4 eager rerun). Cert-mode steady state
   expected ~25-30 s/episode, consistent with the ~12h §12.6 projection, which was written
   to `/tmp/m3_ext_progress.md` at launch.

All preflight gates PASS; the extension campaign starts with no protocol change.
