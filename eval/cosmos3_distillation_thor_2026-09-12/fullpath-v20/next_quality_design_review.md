# Full-path next quality study: CPU-only design review

Review date: 2026-09-12 UTC. **Scientifically reasonable with the amendments below; not execution-ready and not authorization to execute.** The draft asks the right causal question: compare fresh complete-path and last-callback students at the same update budget. V20 demonstrates derivative/optimizer/export feasibility, not quality superiority. Nothing in this review changes V20 or admits its pilot weights.

Only static source, existing JSON receipts and existing control payload hashes were inspected. No training/evaluation entrypoint was imported or run; no RNG was generated/reserved, executable protocol constructed, process scheduled, GPU queried or used, or frozen artifact modified. The sole deliverable is this review in the Compress coordination directory, despite the current shell directory being InstinctFlash.

## 1. Required clarification: formal training population and additive source versions

The phrase “V20/V17 recipe” is ambiguous about the training observations. V17 formal training uses **32 contexts**, whereas V20 explicitly selects `old["training_contexts"][:2]` and replaces their preparation seeds. Keeping V20's two-observation pilot population while merely increasing updates would not reproduce the V17 formal data recipe. Specify the existing V17 32-context membership/order/dataset identity for both treatments, with the intended fresh paired preparation draws; do not add a new dataset or population. Probe selection remains two declared contexts per seed, distinct from saying the training set contains only two contexts.

Evidence: [V20 context selection](/home/ubuntu/InstinctCompress/examples/fullpath_protocol_v20.py:77), [V17 frozen formal protocol](/home/ubuntu/InstinctCompress/results/cosmos3_droid/twostep_v17/training_protocol_v2.json), [V20 frozen pilot protocol](/home/ubuntu/InstinctCompress/results/cosmos3_droid/fullpath_v20/pilot_protocol_v1.json).

Existing code cannot be used unchanged as the new study runner. V20 trainer settings/CLI permit only a one-seed, two-update pilot; its protocol fixes exactly two arms, global initialization/LoRA reservations and two contexts; its paired probe auditor requires exactly two reports and keys them only by gradient scope. Its exporter admits only `mode=pilot`. V19 evaluation assumes old seeds12031/12032, one new student prefix and an11-bank Edge catalog, automatically adding prior trained V14/V16 controls. That is incompatible with four fresh arms and seven banks. New separately versioned orchestration, validation, audit and analysis adapters would be required after separate execution authorization; preserve frozen numerical helpers and sources. Do not disable these guards or relabel old outputs.

Evidence: [trainer settings](/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py:37), [pilot-only preparation](/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py:197), [pair audit](/home/ubuntu/InstinctCompress/examples/probe_cosmos3_fullpath_v20.py:381), [export restriction](/home/ubuntu/InstinctCompress/examples/export_cosmos3_fullpath_v20.py:67), [V19 catalog](/home/ubuntu/InstinctCompress/examples/evaluate_cosmos3_followup_v19.py:75), [independent catalog guard](/home/ubuntu/InstinctCompress/examples/audit_cosmos3_followup_v19_v2.py:200).

## 2. Required RNG design: two genuinely distinct pairs, with an explicit reuse exception

Use a pair identifier distinct from the estimator arm identifier. Within a pair, initialization, context preparation, data ordering, endpoint/SDE draws, score times/noise and probes must agree; across the two pairs, newly generated initialization and training/probe reservations must be distinct. V20's single global `extra_rng` and scope-only probe map cannot establish this automatically. Expand reservations by pair, then reference those reservations from both treatments without sharing mutable roles or optimizers. Retain complete VJP comparisons separately for each pair. Verify initial master tensors and pre-student warmup states agree within each pair; after student1, fake trajectories are expected to diverge because they train against their own evolving student. Do not force fake states or score outputs to remain equal after treatment begins.

Fresh training/probe/export reservations must exclude all previous studies **including V20 itself**, which is naturally absent from V20's own `SEED_PATHS`. Account for expanded seed widths, not just training labels, and do not silently skip an expected reservation source because its filename moved. This review selected no seed values.

Evaluation RNG is a deliberate exception: reuse the historical paired-policy draws for all four new student banks so they remain paired with the existing original controls; retain the independent historical teacher32 draws. Requiring *all* evaluation draws to be fresh/disjoint would contradict control-bank reuse. Record inherited evaluation requests separately from new training/probe/export reservations; never treat intentional inherited draws as newly available RNG.

Evidence: [global role initialization](/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py:223), [LoRA seed access](/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py:93), [prior reservation inputs](/home/ubuntu/InstinctCompress/examples/fullpath_protocol_v20.py:9), [query schedule](/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py:118).

## 3. Control-bank reuse: compatible existing evidence, conditional future admission

The correct three banks are:

| Bank | Existing report | Requests | Actual branches | Contract |
|---|---|---:|---:|---|
| Original SDE2 | `twostep_v17/evaluation_capture_v19/original_same_grid/report.json` |128|256|CFG1, zero padding, clocks1000/500|
| Original UniPC4 | `paired_training_v13/evaluation_capture/original_unipc4/report.json` |128|1024|CFG4, native padding, shift1, clocks999/749/499/249|
| Original teacher32 | `paired_training_v13/evaluation_capture/teacher_unipc32/report.json` |128|8192|CFG4, native padding, shift1, independent draws|

Paths above are relative to `/home/ubuntu/InstinctCompress/results/cosmos3_droid/`. All three report the same original Edge backbone hash `697e50d098feb6c0f6b117dd4e7fa7023260b93a895fc25a27bc38120c1cb30b` and matching recorded native H100 runtime. This review verified **all384 request JSON hashes and all384 referenced NPZ hashes**. Across all128 rows, SDE2/UniPC4 request identities apart from configuration name and original pre-padding preparation identities agree. All three banks have matching normalizer and raw/model recorded-target hashes; teacher seeds differ from paired-policy seeds. Existing V17 independent quality audit is successful. No scores or model outputs were recomputed here.

Crucially, compare **pre-padding** inputs to establish common random inputs. Effective padded tensors, sampler clocks and CFG branch counts intentionally differ between SDE2 and UniPC. Require each bank's own exact contract, not identical effective tensors/clocks across all banks. Original UniPC4 is an operational reference, not a gradient-only ablation; original same-grid SDE2 measures the net effect of distillation at that schedule. Only fresh full versus fresh last isolates the assigned estimator treatment. Preserve original owner protocol/admission identities and do not rewrite bank provenance to the new study.

Evidence: [original bank catalog and ownership](/home/ubuntu/InstinctCompress/results/cosmos3_droid/twostep_v17/evaluation_protocol_v19.json), [pre-padding identity extraction](/home/ubuntu/InstinctCompress/examples/audit_cosmos3_pair_evaluation_v13.py:239), [native versus projected padding checks](/home/ubuntu/InstinctCompress/examples/audit_cosmos3_pair_evaluation_v13.py:245), [existing independent audit](/home/ubuntu/InstinctCompress/results/cosmos3_droid/twostep_v17/quality_independent_audit_v19.json).

This is a favorable compatibility review, **not future reuse admission**: new student weights, runtime, prepared inputs, prompt/tokenizer/metadata, decoded normalization and external Wan identity do not yet exist as measured new-study evidence. Later gates must compare them against these controls. Bind the existing Wan blob/resolution code explicitly before execution, rather than repeating V20's transitive ancestor-only declaration followed by an explicit post-execution closure. Stop on incompatibility; no replacement control capture is included.

## 4. Estimator isolation is sound, with the intended downstream effects retained

V20's helper enables both parameter callbacks and keeps the incoming state differentiable only for `complete_path`; the control detaches callback inputs and enables only callback2. The independent reference calls the released sampler with separate velocity callbacks and explicit gradient context. Reuse this mathematical contract without changing arithmetic, native precision, masks, padding derivatives, conditioning, CFG, score-time distribution, clipping or loss. No new straight-through estimator or inference kernel is warranted.

The training implementation applies the chosen derivative only on student updates; fake updates generate endpoints with scope `none`. Teacher/fake DMD targets are formed under no-grad; teacher CFG4 means one score query with **two branches**, while fake CFG1 means one score query with one branch. The draft's wording “conditional single fake/teacher score queries” should not be read as making the teacher conditional-only. Changing both gradients and teacher guidance would confound the study. Each fake must remain trained against its own student's detached endpoints with the same paired random draws and update cadence. EMA may remain a diagnostic state but is not the evaluation/export selection.

Evidence: [rollout graph boundary](/home/ubuntu/InstinctCompress/instinct_compress/models/cosmos3_fullpath_rollout_v20.py:114), [native independent oracle](/home/ubuntu/InstinctCompress/instinct_compress/models/cosmos3_fullpath_oracle_v20.py:7), [student/fake generation scope](/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py:389), [detached score targets](/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py:439).

## 5. Branch arithmetic is correct; specify audit coverage and total accounting

| Component | Predicted branches |
|---|---:|
| Four formal arms:4×(507×2 generation +64×2 teacher +507 fake)|6596|
| Two paired native probe sets:2×(18 control +22 full)|80|
| Four source/export/cold gates:4×6|24|
| Training/probes/exports subtotal|6700|
| Four new student quality banks:4×128×2|1024|
| **All new work, assuming admitted control reuse**|**7724**|

These are predictions, not usage or authorization. The reused controls contain9472 historical branches; adding them to1024 gives the draft's10496 hypothetical full-recapture cost, which remains excluded. The analysis population is896 requests, only512 newly captured. Keep inherited and new compute distinct.

V17 sentinels mean student1/4/16/64, warmup1/4/16/128, continuing fake1/4/16/315:12 full Adam sentinels per arm, **48 total**, and **2028 raw RF/DMD update field sets** across four arms. Retain all four52-parameter role ownership/state-chain audits and snapshots0/1/4/16/64. Count failed/partial calls if any; the total assumes no extra native passes, retries or replay calls. Source/cold separate processes remain necessary. Equal update count is correctly described as unequal training compute; pilot wall time with hashing/I/O is not a throughput guarantee.

Evidence: [actual cadence/count formulas](/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py:159), [formal sentinel settings](/home/ubuntu/InstinctCompress/results/cosmos3_droid/twostep_v17/training_protocol_v2.json), [V20 measured completion](/home/ubuntu/InstinctCompress/results/cosmos3_droid/fullpath_v20/COMPLETION_HANDOFF.json).

## 6. Quality interpretation: fix the contrast catalog and averaging order

The draft's overall interpretation is appropriate. Make “all four comparisons” unambiguous: four refers to the four primary metrics, whereas the estimator contrast has **two seed-specific comparisons plus one within-episode seed-mean comparison**. Explicitly list the corresponding same-grid/UniPC4 comparisons for both treatments, seed-specific and mean, before outputs are read. This can be expressed as3 estimator contrasts plus12 treatment/reference contrasts, all on the existing seven banks; it requires no new model or experiment. The legacy V19 contrast catalog is not applicable.

Compute K8 energy scores **separately for each seed within each episode**, then average the two scores (or paired score differences) within episode before the5000 episode bootstrap. Do not pool both seeds' draws into K16 or average action vectors before calculating energy score: its nonlinear attraction/spread terms would change the estimand. Retain each seed's results and absolute differences; percentage differences use the fixed full-panel baseline mean, including the same denominator for converted interval endpoints. Preserve raw32×8 actions and all96 existing secondary metrics.

The primary `raw/.../energy_u` scores compare to **recorded commands**, not teacher32. Teacher distribution comparisons remain separate secondary diagnostics. Lower average energy score is not a guarantee of safer action tails or closed-loop success; preserve the existing MSE/spread/teacher diagnostics without adding a selection rule. Reused16-episode development data and two training seeds support exploratory estimator evidence, not generalization, practical equivalence or a training-seed population estimate. An improvement against last-callback with continued UniPC4 regression is not accuracy recovery. No metric cherry-picking, checkpoint selection or automatic extension is justified.

Evidence: [energy-score terms](/home/ubuntu/InstinctCompress/examples/audit_cosmos3_pair_evaluation_v13.py:145), [recorded-target checks](/home/ubuntu/InstinctCompress/examples/audit_cosmos3_pair_evaluation_v13.py:165), [existing score-before-seed-average implementation](/home/ubuntu/InstinctCompress/examples/evaluate_cosmos3_followup_v19.py:557).

## 7. Serving compatibility: additive capability exists, new provenance is still required

Preserve `cosmos3_policy_action_fixed_step`: its public contract and runtime wrapper deliberately require CFG4, four callbacks and eight projected branches. It cannot serve these SDE2/CFG1 students by changing a declaration alone. An experimental additive Flash adapter **already exists**, `cosmos3_policy_action_realtime_v16`, and accepts the complete two-step grid/CFG1 with two projected branches and padding-sidecar validation. Thus the schedule capability is available; this review does not imply the old plugin needs modification or that an entirely new sampler must be implemented.

The V17 packaging gate is nevertheless hardcoded to schema17, old seed/arm names and final64 V17 provenance. A future copied package requires separately versioned new-study provenance admission, preserved padding sidecar/helper identity, merged-native loading and its own full-field/source/cold checks. Native export parity does not by itself certify the Flash serving path. Keep copies `servable=false` until their own qualification; no V20 pilot export is eligible. The earlier442–444ms V17 measurement is a same-schedule historical cost reference, not measured latency/quality for new weights. No Thor screen is requested by this review.

Evidence: [old contract](/home/ubuntu/InstinctCompress/instinct_compress/flash/cosmos3_action_padding.py:35), [old eight-branch wrapper](/home/ubuntu/InstinctCompress/instinct_compress/flash/cosmos3_action_padding.py:78), [additive adapter contract](/home/ubuntu/InstinctFlash/eval/cosmos3_distillation_thor_2026-09-12/realtime-deployment/realtime_adapter.py:16), [sidecar admission](/home/ubuntu/InstinctFlash/eval/cosmos3_distillation_thor_2026-09-12/realtime-deployment/realtime_adapter.py:95), [V17-specific package guard](/home/ubuntu/InstinctFlash/eval/cosmos3_distillation_thor_2026-09-12/realtime-deployment-v17/prepare.py:19).

The unspecified controller deadline, executed action chunk and end-to-end tail budget remain independent blockers to any realtime claim. This review ends here: no seeds chosen, executable protocol produced or follow-on scheduled.

## Reviewed source identities

The following small-file SHA256 identities bind the draft and principal sources inspected; they are a review inventory, not a training protocol or RNG reservation. The mutable coordination file hash is its state at review time.

- `/home/ubuntu/InstinctFlash/eval/cosmos3_distillation_thor_2026-09-12/fullpath-v20/next_quality_study.md` — SHA256 `7f2a3f813940832ec9eb9646a9bc2863f7c8f50a02078a90fcc4b895e81667ff`
- `/home/ubuntu/InstinctCompress/coordination/FLASH_COSMOS_REALTIME.md` — SHA256 `47277cd15a06e8b1f0d5272a6e63d9646c4c49a152831ba15a91972b3b2b5486`
- `/home/ubuntu/InstinctCompress/examples/fullpath_protocol_v20.py` — SHA256 `886f4db49d22f9b28eb09d0b1539b9af61c6b92f20c7f0e9582d6f0210746fcc`
- `/home/ubuntu/InstinctCompress/examples/train_cosmos3_fullpath_v20.py` — SHA256 `e2e85a41fee74e5c3fd9a29c1722837ce1d5556ca6a6a03622713018541eaf5d`
- `/home/ubuntu/InstinctCompress/examples/probe_cosmos3_fullpath_v20.py` — SHA256 `907d2a2538fd6e1315882644027f11bc0c5cb773cfb7c1eccb34458b5bc54aa8`
- `/home/ubuntu/InstinctCompress/examples/evaluate_cosmos3_followup_v19.py` — SHA256 `24d432b2fa6627d0f0981aa7ddc9434662f5f437c5d97f07a529bef16afd5e64`
- `/home/ubuntu/InstinctCompress/examples/audit_cosmos3_followup_v19_v2.py` — SHA256 `a07748abc98daddc9202187548c2b812a0a52aa99357997059b86c4087bd2725`
- `/home/ubuntu/InstinctCompress/results/cosmos3_droid/fullpath_v20/COMPLETION_HANDOFF.json` — SHA256 `79af383fbaf216718453e38a5a5796c6f6d13504b36299900828d09d53c1fa63`
- `/home/ubuntu/InstinctCompress/results/cosmos3_droid/twostep_v17/evaluation_protocol_v19.json` — SHA256 `414887835509a04d583acf16c8fd0436b33961869a3cb59bf18e3d47fe01e663`
