# VLA benchmark driver contract

The orchestrator intentionally does not import every model framework. GR00T, LeRobot, LingBot,
Cosmos and DreamZero have mutually incompatible dependency stacks; merging them changes the
baseline being measured. Instead, each arm names an argv prefix in `arms.template.json`. The
pipeline appends:

```text
--request /absolute/path/to/request.json --output /absolute/path/to/pending-result.json
```

The command is executed directly with `subprocess.run(..., shell=False)`. Exact `${NAME}` argv
items are expanded from the environment; partial shell interpolation, globbing and word splitting
do not exist.

Each arm also declares an immutable driver revision. An exact `${NAME}` revision entry is resolved
when the plan is built and the concrete value is stored in the plan. The result must echo that exact
identity; a different driver revision is rejected before aggregation.

## Request

The request file is the complete plan job plus its digest:

```json
{
  "job_id": "24 hex characters",
  "request_sha256": "64 hex characters",
  "request": {
    "pair_id": "shared by control and treatments",
    "model_id": "registry identity",
    "model": {
      "backbone": "pi05",
      "checkpoint": {
        "id": "model repository or immutable local package identity",
        "revision": "full 40-character revision",
        "derived_from_revision": "parent model revision"
      }
    },
    "dataset": {"id": "...", "source": "...", "revision": "..."},
    "suite": {
      "id": "...",
      "kind": "contract | latency | open_loop | closed_loop",
      "seed_strategy": "fixed | increment_until_stable",
      "seed_max_attempts": 100,
      "required_metrics": ["..."]
    },
    "task": "task identity",
    "requested_seed": 50100,
    "repeat": 0,
    "arm": {
      "id": "accelerated",
      "role": "treatment",
      "operating_point": {"name": "candidate", "tier": "NUMERIC"}
    },
    "measurement": {"warmup": 3, "iterations": 30}
  }
}
```

The driver must seed model noise before every paired inference. `increment_until_stable` is for
RoboTwin setup failures: try the requested seed, then increasing seeds, and report the one actually
used. Both arms must resolve to the same seed or the pair is rejected.

TF32, autocast and quantization controls must be scoped to model inference. They may not leak into
the simulator process. A split client/server driver naturally provides this isolation; an in-process
driver must use a scoped context and restore global settings before environment stepping.

## Result

Write one JSON object atomically to the requested pending output path:

```json
{
  "schema_version": 1,
  "job_id": "echo from request",
  "request_sha256": "echo from request",
  "status": "completed",
  "resolved_seed": 50101,
  "metrics": {
    "finite": true,
    "latency_ms": [72.1, 72.3, 71.9],
    "success": true,
    "action_digest": "sha256 over a canonical action trace",
    "action_values": [0.1, 0.2]
  },
  "provenance": {
    "model_revision": "exact checkpoint revision from the request",
    "driver_revision": "git revision or immutable release identity",
    "environment_fingerprint": "sha256 of the driver's lock/interpreter/CUDA description",
    "synthetic": false
  }
}
```

Only metrics listed in `required_metrics` are mandatory for a job. `action_values` must be a flat,
canonical float list when a NUMERIC open-loop gate is requested. Closed-loop drivers may emit only
an `action_digest`; success non-inferiority is their deciding quality gate.

Drivers should hash actions after conversion to a declared common dtype and contiguous layout. The
hash convention belongs in `driver_revision`; changing it is a driver revision change.

`synthetic=true` means the driver executed no model: it is reserved for `reference_driver.py`
(CI orchestration checks) and `replay_driver.py` (recorded closed-loop outcomes fed through the
gate path to validate the report stage against the frozen `certify()`). Reports containing it
are non-reportable unless the caller explicitly uses `--allow-synthetic`; a replay validates the
pipeline and is never fresh benchmark evidence.

For the implemented LingBot-VA paused-simulation bridge and its additional scene/identity
checks, see [ROBOTWIN.md](ROBOTWIN.md).

### Closed-loop action evidence

Standard reports include `comparisons[].closed_loop_actions` alongside the statistical
`closed_loop` quality gates. Each model/suite entry records expected/observed pairs,
matching/different trajectories, per-pair steps and success, and missing pair IDs.
`PASS` requires positive executed step counts, finite actions, matching scene identity,
matching counts and matching action SHA-256 digests for every expected pair.
Missing evidence is `INCOMPLETE`; observed differences are `FAIL` even if other pairs
are missing. Unknown digest protocols are `NOT_APPLICABLE`.

This is observational evidence, marked `quality_gate=false`: it does not enter the
success non-inferiority gate or imply that a lossy variant with different actions has
lower task success. Screening remains non-reportable regardless of matching actions.
Likewise, a quality gate passing does not certify bit-exactness; inspect both fields.

Encoding is explicit and protocol-specific. `wan-va-libero-paused-v1` hashes each
executed 7D action's native dtype string and contiguous bytes. The existing
`wan-va-robotwin-paused-v1` hashes 16D controller values converted to big-endian FP64;
its result does not attest the model tensor's original dtype. Neither protocol proves
hidden-state equality, equivalence on other inputs, or equivalence on another device.
New adapters must define their digest encoding before the reporter recognizes them.

When reanalyzing a completed run, use `report --output <new-path>` to preserve its
original report. `execution_pipeline_sha256` identifies the frozen execution code;
`analysis_pipeline_sha256` identifies the code producing this report. Reanalysis
reads the existing evidence and does not rerun models or rewrite the frozen plan.
