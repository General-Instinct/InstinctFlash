# Genuine two-step training handoff

The native and cuDNN serving-cost screens do not create trained two-step weights.
Source inspection of the existing Compress V16 CLI found that setting the
`RealtimeConfig.steps` dataclass alone is insufficient:

- `train_cosmos3_realtime_v16.py:73` constructs `RealtimeConfig` without steps;
  line 408 constructs the student `DeploymentGrid(1, ...)` explicitly.
- Arm generation, snapshot/manifest fields and execution reports still declare
  one step/`complete1`; expected callback and branch accounting must follow the
  actual student grid.
- `pilot_equivalence` at line 835 compares to the old one-step fixed rollout.
  A two-step pilot needs an independent full native `[1,.5,0]` endpoint and
  all-parameter VJP reference. Changing `prefix_steps` on an old longer schedule
  is not automatically the same complete deployment contract.
- `audit_cosmos3_realtime_training_v16.py` hardcodes `complete1` and
  `DeploymentGrid(1, ...)`; the audit must independently enforce the new grid.
- `export_cosmos3_realtime_v16_v2.py:406` still uses `DeploymentGrid(1, ...)`.
  Persisted native sampler config, padding contract, source export and separate
  cold-load checks all need to agree on both `[1000,500]` callbacks.

Teacher/fake score evaluations also use `DeploymentGrid(1, ...)` to describe a
single score query. Those are separate from the student's two-step deployment;
do not replace every occurrence of `1` mechanically.

Use a separate prospective study with fresh original student/fake roles, both
training seeds, bounded native optimizer pilot, explicit gradient scope, and
full native endpoint/VJP/Adam/export audits before a full training queue. Keep
V16 one-step protocols, weights and published receipts frozen. The Flash
serving adapter already supports the complete two-step schedule; CPU native
endpoint tests and original-weight Thor screens do not replace a trained-model
qualification.

H1004–7 scheduling remains Compress-owned. No new two-step or Nano DMD training
is claimed here, and Nano mechanical qualification weights remain excluded.
