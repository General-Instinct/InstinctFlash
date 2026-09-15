# Runtime regression

The Thor Cosmos suite compares NVIDIA's eager DROID service with InstinctFlash
in three fresh processes: reference A, candidate, reference B. It preserves the
checkpoint's BF16 parameters, four UniPC steps, CFG 3 and full 32×8 action chunks.
Recorded camera frames, varying states, two prompts, continuous requests and
resets are identical across arms. Cold requests are retained; each prompt is
warmed before steady-state timing.

```bash
# Use the existing Cosmos environment on an idle Thor.
python -m benchmarks.regression.run_cosmos /absolute/path/to/new-results
```

Each run saves the request protocol, source and checkpoint identities, per-call
latencies, complete action arrays, execution statistics and an A/B/A comparison.
The runner holds `/tmp/thor_gpu.lock`; foreign compute processes are checked
before loading and before each request. It does not install dependencies or
change GPU clocks.

Before deploying a benchmark bundle that uses compiled extensions, inventory the
source and required native artifacts together:

```bash
python -m benchmarks.regression.runtime_bundle /path/to/bundle /path/to/manifest.json \
  --require-binary 'serving/flash_rt/flash_rt_kernels*.so'
python -m benchmarks.regression.runtime_bundle /path/to/bundle /path/to/manifest.json --verify
```

Missing extensions or changed source/binary inventories fail the check without
loading CUDA. Interpreter compatibility and actual kernel execution still need
their own preflight and E2E checks. Checkpoint weights and external vendor
environments retain separate manifests.

The gate requires finite, byte-identical action chunks and stable references.
A p50 increase over 5% or p95 increase over 10% fails performance regression.
Reference p50 drift over 5% or contention produces an inconclusive performance
verdict and a nonzero exit. Missing metadata, changed source files, mismatched
protocols or action hashes invalidate the run. This tests action regression;
it does not certify closed-loop task success or qualify FP8.

## Automated runs

`runtime-regression.yml` runs the artifact-gate tests on relevant pushes and PRs.
A scheduled GPU run can use the committed checkout through SSH:

```bash
python benchmarks/regression/deploy_thor.py \
  --host user@thor \
  --remote-root /home/user/ifl-regression \
  --python /home/user/cosmos-framework/.venv/bin/python \
  --output-root /absolute/path/to/local-results
```

The deployer archives `git HEAD`, runs that isolated source snapshot on Thor,
and retrieves receipts. `latest.json` identifies the latest run, including its
exit status; an occupied GPU is never marked as a pass. Run once with
`--accept-baseline` to accept a reviewed passing run. Later runs also compare
candidate/reference latency ratios with that fixed baseline, so losing an
existing speedup fails even if the candidate still beats upstream. The baseline
is never silently replaced. Changed protocols require explicit rebaselining. Existing SSH access and
cached checkpoint dependencies are required. It never pushes, fetches or executes
uncommitted changes.

`run_cosmos --candidate conditioning` explicitly tests the optional native Thor
conditioning cache. It requires successful source admission, real-input byte
checks and graph replays; a safe runtime fallback cannot count as a passing cache
benchmark. This does not change the accepted nightly baseline.

For optimization experiments, `run_cosmos --candidate pointwise` enables the
pointwise candidate and `--candidate graphs` additionally tries bounded layer
CUDA graphs. `--candidate reuse` combines pointwise fusion with the existing
Nano action-only constructor (Edge uses pointwise alone). These overrides are recorded through backend statistics; ordinary
nightly runs use the default Runtime configuration.

The `systemd/` templates schedule a nightly run at 03:00 UTC with up to 15 minutes
of jitter. Configure `~/.config/instinctflash/thor-regression.env` with
`REGRESSION_PYTHON`, `THOR_HOST`, `THOR_RESULTS`, `THOR_PYTHON` and `LOCAL_RESULTS`;
copy the units to `~/.config/systemd/user/`, then enable the timer with
`systemctl --user enable --now instinctflash-thor-regression.timer`.
The template assumes the checkout is `~/InstinctFlash`. Inspect the journal and
`latest.json` for failures; no external messages are sent.

## Native numerical screen

```bash
python -m benchmarks.regression.run_cosmos_numeric /absolute/path/to/new-results
```

On the screened Thor stack this runs Edge and Nano, each with a fresh native
BITEXACT / native NUMERIC pair through `Runtime.from_pretrained`. Both arms use
guarded conditioning reuse; only the numerical arm enables cuDNN BF16 attention.
It records six warmup and ten measured full 32×8 action chunks per arm, checks
source/protocol agreement, finiteness, actual backend execution and cache replay,
and reports p50/p95, speedup and action differences. Use `--family edge` or
`--family nano` for one model; `--compare-only` rechecks existing artifacts. The default performance gate requires
at least 1.1× p50 speedup (`--min-speedup`); an optional
`--max-abs-action-delta` adds an explicit action-delta regression bound. Gate
failure returns a nonzero exit. That bound does not certify task success.

These reports are **SCREEN** evidence, with no closed-loop quality certificate
or task-success margin. They remain separate from the strict nightly byte-equality
gate. Quantization and fewer denoising steps are not enabled by this command.

## Dynamic prediction reuse

The shared step-cache controller and public selection have CPU regressions in
`tests/test_step_cache.py`, `tests/test_step_cache_policy.py` and
`tests/test_dreamzero_step_cache.py`. Public execution snapshots are covered by
`tests/test_runtime_telemetry.py`. The bounded
[Thor integration study](../../eval/dynamic_step_cache_integration_2026-09-14/README.md)
compares native/FP8 arithmetic with fixed/dynamic scheduling and independently
audits the actual prediction counts, actions and retained native-state hashes.
Task-quality acceptance remains a separate gate for every checkpoint/profile.
