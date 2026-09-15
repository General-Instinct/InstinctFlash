# DreamZero construction reproducibility diagnostic

Status: prepared on Thor; native and FP8 **file-only preflights passed** (154 bundle files). No seeded-construction inference has run yet. Synthetic comparator controls pass; these are not model results.

The previous frozen-source rerun reproduced the newer native actions, but differed from the earlier native run. Per-prediction seeds were fixed; construction seeds were not controlled. This experiment tests that possible source of variation without claiming it is the cause.

The isolated Thor bundle is `/home/guanming/ifl_eval/thor_precision_completion_20260909/dreamzero-seeded-bundle-v2`. It copies the preserved `dreamzero-current-source` implementation. One preexisting dangling link, `instinctflash/runtime/lingbot_worker.py`, is explicitly excluded in `staging.json`; it is unrelated to DreamZero. Failed bundle v1 is retained. This is a diagnostic of that preserved implementation, not qualification of later production changes.

After the active VA campaign releases Thor, run two distinct processes for each precision with the bundle launcher:

```bash
python run_dreamzero_seeded_construction.py native repeat1
python run_dreamzero_seeded_construction.py native repeat2
python run_dreamzero_seeded_construction.py fp8 repeat1
python run_dreamzero_seeded_construction.py fp8 repeat2
```

Run from the Thor bundle directory using the installed `dz_env` Python. The launcher validates all bundle hashes and source inventory, acquires `/tmp/thor_gpu.lock` without waiting, fixes the import environment, and refuses to overwrite logs. Preflight-only mode imports no model and uses no GPU.

For each precision compare its two output directories with `compare_dreamzero_seeded_construction.py PRECISION LEFT RIGHT OUTPUT`. The comparator requires distinct process/run IDs, matching construction RNG, source/input/probe hashes, checkpoint, Torch/device/numerical settings, original schedule, completed six-chunk histories and intact artifacts. It reports differences rather than treating a reproducibility mismatch as a successful equivalence check.

The probe fixes construction seed 9173 before Runtime creation, hashes full persistent model state, and preserves the original per-prediction seeds and two three-cycle recorded-input episodes. Source hashes must remain equal before and after execution. Input/state/action archives are bound to completion receipts. This does not cover nonpersistent buffers, native binaries, physical task success, or general BITEXACT equivalence. State hashing perturbs timing, so this experiment cannot supply latency claims.

[Bundle manifest](dreamzero-seeded-bundle-v2-manifest.json) · [Probe](probe_dreamzero_seeded_construction.py) · [Comparator](compare_dreamzero_seeded_construction.py) · [Synthetic controls](test_compare_dreamzero_seeded_construction.py)
