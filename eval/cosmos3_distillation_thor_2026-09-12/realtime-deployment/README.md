# Experimental V16 student deployment

`realtime_adapter.py` adds an explicitly registered experimental backbone:
`cosmos3_policy_action_realtime_v16`. It uses merged native Cosmos checkpoints,
NVIDIA's fixed-step SDE sampler and Compress's existing zero-padding helper.

Supported declarations use complete grids `[1,0]`, `[1,.5,0]` or
`[1,.75,.5,.25,0]`, literal CFG1 or CFG4, timestep scale 1000 and native DROID
32x8 outputs. NFE must match the entire grid. A two-step intermediate exit from
a four-step grid is rejected. The padding sidecar must pin the installed helper,
and referenced checkpoint payloads must exist inside the package.

```python
from instinctflash import Runtime, register
from realtime_adapter import BACKBONE, Cosmos3RealtimeAdapter

register(BACKBONE, Cosmos3RealtimeAdapter)
# The package must explicitly declare this backbone and its exact schedule.
# Unqualified study packages remain servable=False and require strict=False.
api = Runtime.from_pretrained(package, strict=False, precision="native",
                              placement="in_process")
```

The wrapper checks the loaded service's guidance/step count, projects padding
at each native query/transition, and checks one preparation plus the expected
actual CFG branch count per request. It inherits checkpoint-native prompt,
image/state processing and decoding. Historical backbones and production
numeric admission are unchanged; FP8 remains rejected by this experimental
fixed-step adapter.

CPU integration: **14 tests passed**, including all six grid/guidance pairs
through the actual native sampler and zero-padding helper. Outputs are exactly
equal to the producer's `rollout_deployment` on its tiny CPU model fixture.
Tests also check exception restoration, metadata, invalid grids/guidance and
unchanged historical contract rejection. Source identities are recorded in
[cpu_validation.json](cpu_validation.json).

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/home/ubuntu/InstinctFlash:/home/ubuntu/InstinctCompress:/home/ubuntu/InstinctCompress/tests:/home/ubuntu/InstinctFlash/examples/cosmos3_policy:/home/ubuntu/cosmos-framework \
/home/ubuntu/lingbot-vla-repo/.venv/bin/python -m pytest -q \
  eval/cosmos3_distillation_thor_2026-09-12/realtime-deployment/test_realtime_adapter.py
```

The CPU checks are separate from the actual Edge student Thor runs below.
Full-model producer-helper versus public Runtime parity and deployed task
quality remain pending. Nano student deployment needs its own qualification.

## Thor paired qualification

`run_pair.py CHECKPOINT FRESH_OUTPUT --fixture FIXTURE` serializes fresh native
and cuDNN processes under the Thor GPU lock. Both use the package's complete
SDE grid and literal CFG, with six warmup and ten measured requests. The runner
explicitly allows unqualified packages; it does not promote them.

`benchmark_candidate.py` records actual first-request branch clocks, total
sampler/branch counts, padding restoration, full finite actions and source
hashes including the V16 plugin. cuDNN is installed only as an explicit offline
experiment after pinned vendor-source checks. The measurement accepts Edge/Nano
layer counts; that is not evidence that either trained package has passed.

`compare.py OUTPUT` validates the matched manifests, protocol, source hashes,
complete clocks, branch budgets and actions before reporting attention speedup
and action drift. It recomputes latency percentiles and invalidates an old pass
before checking new data. Thirteen synthetic receipt tests passed, including
rejection of incorrect branches/clocks, stale sources, changed manifests,
nonfinite outputs, missing padding restoration and false latency summaries:

```bash
# Use the CPU-only PYTHONPATH/environment above.
python -m pytest -q eval/cosmos3_distillation_thor_2026-09-12/realtime-deployment/test_comparison.py
```

The actual final-online64 Edge student results below use complete SDE1.
Original-weight UniPC budgets remain in their separate study directories.

## Final-online64 package preparation

`prepare.py EXPORT_ROUNDTRIP FRESH_DESTINATION --model-id STUDENT_ID` accepts
only the four declared Edge SDE1 CFG1/CFG4 final-online64 arms (both seeds).
It checks the producer export gate, hashes of linked evidence and snapshots,
separate-process cold checks and complete native inventory, then copies the
native checkpoint into a fresh directory. It adds the experimental backbone
and pinned zero-padding declaration with `servable=False`, verifies the new
package, and rechecks original files and evidence. Frozen producer exports
remain unchanged. Nano mechanical pilot exports are not accepted here.

Fourteen gate tests passed, covering all four arms and rejection of pilot,
EMA, wrong-grid/guidance/padding and incomplete cold evidence. These gate
tests are supplemented by the actual copied-package receipts below.

```bash
# Use the CPU-only PYTHONPATH/environment above.
python -m pytest -q eval/cosmos3_distillation_thor_2026-09-12/realtime-deployment/test_prepare.py
```

## Paired quality request seeds

The normal Runtime seed initializes the service's NumPy stream; it is not the
actual native seed used by the first request. `request_seed.py` provides
`predict_with_request_seed(runtime, observation, seed)` for an isolated offline
quality worker. It temporarily selects native deterministic-seed mode, observes
exactly one generation call carrying the specified request seed, then restores
the normal config, generation binding and unchanged RNG stream, including after
native failures. No public default changes.

Seven CPU tests passed using the actual vendor `_next_seed` method extracted
from its source (GPU loaders are not imported). They verify direct seed pairing,
stream/config restoration and invalid-input rejection. This does not yet prove
full-model noise, conditioning, endpoint or decoded-action equivalence; those
must be captured on actual exported students. Do not use this helper concurrently
with other requests or include its instrumentation in published latency results.

## External WanVAE dependency

Cosmos resolves an external `Wan2.2_VAE.pth` in addition to the checkpoint's
native inventory. [Thor preflight](thor_wanvae_preflight.json) records the actual
2,818,839,170-byte cached blob, independently hashed to the producer's disclosed
V11 anchor. This does not retroactively expand any older experiment's inventory.

The new benchmark uses `runtime_assets.audit_vae_load` during model construction.
It observes the actual `easy_io.load` call, checks content before deserialization
and again after loading, records the resolved path, then restores the loader.
Exactly one matching external load is required. It does not redirect resolution
or substitute weights. The paired comparison requires matching external assets.
Four new observer tests plus thirteen paired-result tests passed. The successful
V16 Thor runs below observed and verified the actual external load in each arm.

## Actual final-online64 Edge student on Thor

All four CFG1/CFG4 students (seeds12031/12032) passed the producer's source/native/cold export gate,
then were copied to separate new packages. All 35 original files and linked evidence were
verified before copying and rechecked afterward. The package remains
`servable=False` and has no task-quality certificate.

| Student (complete SDE1) | Attention | p50 ms | p95 ms |
|:--|:--|--:|--:|
| CFG1, seed12031, online64 | Native BF16 | 409.70 | 411.41 |
| CFG1, seed12031, online64 | BF16 cuDNN | 271.35 | 272.62 |
| CFG4, seed12031, online64 | Native BF16 | 705.05 | 706.45 |
| CFG4, seed12031, online64 | BF16 cuDNN | 436.01 | 436.37 |
| CFG1, seed12032, online64 | Native BF16 | 409.32 | 412.25 |
| CFG1, seed12032, online64 | BF16 cuDNN | 270.24 | 270.48 |
| CFG4, seed12032, online64 | Native BF16 | 709.05 | 713.47 |
| CFG4, seed12032, online64 | BF16 cuDNN | 435.75 | 436.39 |

The CFG1 matched attention pair is 1.51x faster. Maximum absolute decoded-action
drift is 0.0234375, mean 0.00245292 over 16 full 32x8 outputs. All 16 requests used
one actual native timestep 1000 and one CFG branch, with zero-padding execution
and restored hooks. External VAE, manifest, source and fixture identities match.
This is an attention SCREEN on an actual trained student, not task accuracy or
closed-loop realtime evidence. Both training seeds remain in the quality study.

The CFG4 attention pair is 1.62x faster, with max/mean absolute action drift
0.109375/0.01320346. It executes two branches at timestep 1000 per request.
CFG1's cuDNN p50 is 37.8% lower than CFG4's, but that comparison involves different
trained weights and guidance, and does not establish their relative quality.

[CFG1 raw comparison and receipts](receipts/cfg1-seed12031/comparison.json) ·
[CFG4 raw comparison and receipts](receipts/cfg4-seed12031/comparison.json).
The first attempt placed the VAE observer only around Runtime construction;
model loading is lazy until `reset`, so that attempt exited before actions.
Its failed JSON/log are retained in `receipts/cfg1-seed12031-failed-v1`. The
successful frozen scripts are `/home/guanming/ifl_eval/cosmos_distill_20260912/realtime-deployment-v2`
and the outputs are `results-v16-cfg1-seed12031-v2` and
`results-v16-cfg4-seed12031-v2` under the same root.

The second-seed pairs passed the same full execution and identity checks:
[CFG1](receipts/cfg1-seed12032/comparison.json) and
[CFG4](receipts/cfg4-seed12032/comparison.json). Attention speedups are 1.515x and
1.627x respectively; max action drifts are 0.015625 and 0.109375. CFG1's two
cuDNN p50 values are 270.24–271.35ms and CFG4's are 435.75–436.01ms. These are
separate ten-request benchmark samples, not confidence intervals. Seed12032
runs occurred later on the same Thor; source/protocol/manifest pairing is
validated within each native/cuDNN pair.
