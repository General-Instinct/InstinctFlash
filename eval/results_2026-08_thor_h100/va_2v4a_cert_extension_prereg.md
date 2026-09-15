# Pre-registration: LingBot-VA 2V/4A operating-point certificate, evidence extension to n=1200

Declared: 2026-08-25, BEFORE any new episode was run. Author: the va-cert-extension session
(guanming@general-instinct.com's box). This document freezes the design, the data-handling rules,
and the single analysis. Nothing below changes after data collection starts.

## 1. Question

Is the untrained 2V/4A operating point (teacher weights, `--degrade-nfe 2,4`) non-inferior to the
teacher NFE (25 video / 50 action) on RoboTwin 2.0, at the pre-registered margin of **-0.05
absolute task success**, under the repo's pre-registered paired analysis
(`eval/lingbot_va_robotwin/certify_operating_point.py`, unmodified)?

## 2. The disclosed interim look (n=600) — reported, not quoted as evidence

`cert4a_teacher` vs `cert4a_student` (2026-08-06, 50 tasks x 12 episodes, paired):

```
  control   cert4a_teacher           0.9217
  treat     cert4a_student           0.8983
  paired episodes 600   tasks 50
  delta -0.0233   discordant 28 for / 42 against (11.7%)
  exact McNemar two-sided p      0.1196
  McNemar-SE normal 95% CI       [-0.0507, +0.0040]
  bootstrap 95% CI, iid episodes [-0.0517, +0.0033]  <-- primary (most conservative)
  bootstrap 95% CI, TASK cluster [-0.0517, +0.0033]
  NOISE FLOOR (cert50_teacher, 500 pairs): discordance 19+21 = 8.0%, delta -0.0040
  margin declared -0.050   primary lower bound -0.0517   slack -0.0017
  VERDICT: NOT ESTABLISHED
```

Full stdout archived at `/tmp/va_cert4a_interim_n600.txt` (deterministic reproduction; the tool's
bootstrap seed is fixed). Per the protocol text printed by the tool itself: the margin is NOT to
be relaxed; reject the operating point or add evidence. This extension adds evidence. The n=600
look is disclosed here and is superseded by the single pooled analysis in §7; it will not be
quoted as a standalone result.

Honest defect disclosed with it: the cert4a arms ran with the stock (unpinned) upward seed
search, whose expert gate is nondeterministic across runs. Cross-checking the two arms' client
logs shows **18 of 50 tasks diverge; 108 of the 600 index-pairs compare different scenes** across
arms (worst: place_can_basket 12/12). This is unbiased with respect to the arm (gate acceptance
does not depend on the policy under test) but adds pure noise to the discordance. The existing
600 pairs are kept exactly as recorded — pairing by `episode_id`, as the tool defines — and the
extension closes this failure mode by pinning (§4).

## 3. Design of the extension

* **+600 new pairs: 12 new episodes per task x 50 tasks, both arms**, pooled with the existing
  600 for a single primary analysis at n=1200.
* Arms are identical in configuration to the cert4a runs (reconstructed from
  `/tmp/chain_4a.sh` and the `cert4a_srv_*.log` banners):
  - **Control (teacher)**: `serve_variant.py --config-name robotwin --no-fsdp --no-empty-cache
    --no-debug-dump --conditioning-prefill --ring-kv` (banner: fsdp_elision,
    allocator_churn_elision, debug_dump_elision, conditioning_prefill, ring-kv,
    generic-passes(default)). Teacher NFE 25/50. No conv-layout, exactly as cert4a.
  - **Treat (student)**: same + `--degrade-nfe 2,4`.
  - Checkpoint `/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin`, server env
    `/home/ubuntu/.venv-lingbot` (flash-attn NOT installed — verified before launch), client env
    RoboTwin `.venv`, `IWM_FA_SHIM` import-only stub, launched via
    `torch.distributed.run --nproc_per_node 1`, websocket ports 29056+.
  - Client protocol unchanged: `--task_config demo_clean --seed 0` (st_seed = 10000*(1+seed) =
    10000), `instruction_type='seen'`, `--test_num 12`, driven by the unmodified
    `run_paired.sh` (run name `cert4a_ext` -> result dirs
    `/home/ubuntu/iwm_results/cert4a_ext_teacher` and `..._student`).
* **GPU plan**: teacher on GPUs 0-3 (ports 29056-59), student on GPUs 4-6 (ports 29060-62).
  **GPU 7 stays free** for sibling work throughout.

## 4. Scene set (frozen before any run)

The new episodes are **pinned** via the client's `IWM_SEED_CACHE` mechanism to
`/home/ubuntu/iwm_seeds/cert4a_ext/<task>.json`, frozen before this document was completed.
`_manifest.json` sha256: `d8465864d2616886048ca06a19b1b27bf8e50c282737a5edcd1140e6113cac79`.

Rule (declared, mechanical, no discretion): for each task, the extension seeds are the **first 12
seeds in the baseline50 accepted-seed cache (`/home/ubuntu/iwm_seeds/baseline50`) strictly greater
than the maximum scene seed used by either cert4a arm on that task**. Consequences:

* Disjoint by construction from every scene in the existing 600 pairs, in both arms.
* Both extension arms evaluate the SAME pinned scenes — the §2 divergence defect cannot recur.
* All seeds come from the same acceptance process (the st_seed=10000 upward search) as the
  original 600, so the extension is a continuation of the same scene distribution, not a new one.

Gate nondeterminism on replay: if the expert gate rejects a pinned seed on this run (observed
occasionally on this harness), the client skips it — it is never silently replaced by an
unpinned seed. That episode is then absent from that arm, becomes unpaired at analysis, and is
**dropped and reported by the tool, never imputed** (its pre-registered behaviour).

## 5. Episode identity and pooling (declared before running)

The client indexes episodes chronologically from 0 within a run, so the extension's raw
`episodes.jsonl` carries indices 0..11. Pooling rule:

* Original 600 per arm: `episode_id` unchanged (`<task>/0..11`).
* Extension episodes: `episode_id` := `<task>/(12+j)` where **j is the position of the episode's
  scene seed in that task's pinned 12-seed list**. The per-episode scene seed is read from the
  client log's `current seed:` line sequence (index k in the mp4 filename = k-th accepted seed in
  the log), cross-checked against `emit_episodes.py` output counts. Keying by pinned position
  (not by chronological index) keeps the pairing correct even if one arm loses an episode to a
  gate rejection.
* Pooled dirs: `/home/ubuntu/iwm_results/cert4a_pool_teacher/episodes.jsonl` and
  `..._pool_student/episodes.jsonl` = original + remapped extension records. No record is edited,
  dropped, or imputed in pooling; the only transformation is the declared index remap.

This is data plumbing only. The analysis tool is not modified in any way.

## 6. No peeking

No success rates, deltas, or per-arm aggregates are computed or read for the extension episodes
until both arms are complete and pooled. Progress monitoring during the run is by **episode
count only** (`/tmp/va_cert_ext_progress.md`). The analysis in §7 runs exactly once.

## 7. The single pre-registered analysis

Command, frozen verbatim (the tool is the pre-registered analysis and is NOT modified):

```
/home/ubuntu/.venv-lingbot/bin/python /home/ubuntu/InstinctWM/eval/lingbot_va_robotwin/certify_operating_point.py \
  --control cert4a_pool_teacher --treat cert4a_pool_student \
  --repeat cert50_teacher \
  --margin -0.05 \
  --label "2V/4A vs teacher NFE, pooled n=1200 (pre-registered extension of the n=600 interim)" \
  --out /home/ubuntu/iwm_distill/thor_column/va_2v4a_certificate_final.json
```

* **Margin: -0.05, unchanged.** Not to be relaxed under any outcome.
* **Verdict rule: the tool's own** — non-inferior iff the most conservative of its three lower
  bounds (McNemar-SE normal, iid bootstrap, task-cluster bootstrap) is strictly greater than
  -0.05. Reported exactly as printed: NON-INFERIOR or NOT ESTABLISHED.
* **Noise floor**: `--repeat cert50_teacher` — an independent run of the control (teacher-NFE)
  configuration under the same variant chain, sharing 500 episode ids with the pooled control.
  The prior certificate protocol (`run_rc_validation.sh`) used a repeat arm the same way. Per the
  tool, the floor is informational and cannot overturn the interval verdict.
* Full stdout is saved to `/home/ubuntu/iwm_distill/thor_column/va_2v4a_certificate_final.txt`,
  JSON via `--out` as above.

## 8. Power, and what happens on each outcome

At true delta -0.0233 and 11.7% discordance, pooling to n=1200 gives an expected primary lower
bound ~= -0.043 and **P(pass) ~= 75-80%**. This is a real chance of failure and is accepted.

* **NON-INFERIOR** -> the certificate closes; the README's operating-point row cites it.
* **NOT ESTABLISHED** -> reported exactly so, with the n the observed pooled delta and
  discordance would require (post-hoc, clearly labelled). No further extension without a new
  pre-registration; the default per protocol text is rejection of the operating point.
* Harness failures (crashed task, dead server) are fixed and the affected task re-run in full on
  the affected arm's extension dir (wiping only that task's extension artifacts first); harness
  repair never touches analysis or protocol. All such events are logged in the progress file.

## 9. Preflight gates (harness integrity, run before the arms; results appended below)

1. `check_prompt_parity.py` — live T5 vs baked training embedding, must be bit-exact
   (closes the documented 22.7h silent-failure class).
2. flash-attn absent from `/home/ubuntu/.venv-lingbot` (baseline environment invariant).
3. All 7 servers up on their declared ports; 8th GPU untouched; fleet otherwise idle at launch.

---

### Preflight results (appended after gates ran, before any episode)

1. Prompt parity (2026-08-25, GPU 0): **PASS** — positive prompt and CFG-negative both
   `max|live - train| = 0.000e+00`, "PROMPT PARITY: PASS -- serving reproduces the training
   text conditioning."
2. flash-attn in /home/ubuntu/.venv-lingbot: **ABSENT** (pip list | grep -i flash empty) —
   baseline environment invariant holds.
3. Fleet: all 8 GPUs at 0 MiB / 0% before launch; servers and 7/7 port check recorded in
   /tmp/va_cert_ext_progress.md at launch boundary.
