# RoboTwin closed-loop quality evaluation

The first supported bridge is **LingBot-VA (wan_va) × RoboTwin**, using the upstream
raw reset → infer → observed-history commit protocol. The simulator pauses while the
policy computes. Results measure task success; they do **not** certify realtime execution.
No distillation or reduced-step configuration is enabled automatically.

## Environment and identity

Use separate Python environments for the simulator and model server. The simulator needs
RoboTwin's assets/dependencies plus LingBot-VA's `evaluation/robotwin` client, numpy,
msgpack, and websockets with the synchronous client API. Run commands from this repository.
Set absolute `ROBOTWIN_ROOT` and `LINGBOT_ROOT` paths. The local RoboTwin git revision must
match the selected registry's `robotwin2_eval` revision. Local source/config bytes and all
asset bytes are additionally hashed; changed assets or code invalidate a frozen campaign.
The first implementation rehashes assets per episode, which can be expensive.

For an exploratory checkout, create an explicit alternate registry recording that revision;
pass it to `--make-plan --registry`. Do not relabel exploratory runs as historical results.
Keep simulator hardware, rendering stack and installed dependencies fixed across arms.
Initial robot state and all three RGB observations must match the frozen scene exactly;
a mismatch aborts the episode rather than silently changing the comparison.

## Start identified policy servers

On each model host, use a pinned Hugging Face snapshot containing the actual checkpoint:

```bash
export LINGBOT_ROOT=/absolute/path/to/lingbot-va
export LINGBOT_CKPT=/absolute/path/to/models--robbyant--lingbot-va-posttrain-robotwin/snapshots/8c9dea8abbc5c91cc9e18bc3264b8915083bbe70
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29601 /path/to/model/python \
  eval/lingbot_va_robotwin/serve_variant.py --config-name robotwin \
  --port 29061 --save_root /absolute/path/to/server-output \
  --benchmark-receipt /absolute/path/to/control.receipt.json
```

The receipt is written after model loading and socket binding. It records weight bytes,
execution settings, installed passes, source bytes and environment identity. Copy it to
the evaluation host. Use a separate port/device/receipt for a candidate. For baseline A/A,
two arms may use the same endpoint sequentially. The server accepts only one active client
and acknowledges the episode noise seed on every reset. Connection/inference timeouts and
identity mismatches fail the job, leaving incomplete evidence rather than a quality pass.

This entrypoint supports the existing PyTorch `serve_variant` path, including compatible
optimization flags, on its supported hardware. Native Thor engine serving still needs this
identity/reset protocol implemented; it is not made compatible merely by supplying a URL.
Block-head checkpoints are not supported by this first server bridge.

## Freeze scenes, then run the paired plan

Create an arms file using the actual receipt for each endpoint. For example, baseline A/A:

```python
import json
from pathlib import Path

root = Path('/absolute/path/to/InstinctFlash')
identity = json.loads(Path('/absolute/path/to/control.receipt.json').read_text())
arms = {'schema_version': 1, 'control_arm': 'baseline', 'arms': []}
for name, role in [('baseline', 'control'), ('repeat', 'treatment')]:
    arms['arms'].append({
        'id': name, 'role': role,
        'driver': {
            'command': ['/path/to/simulator/python', str(root / 'benchmarks/vla/robotwin_driver.py')],
            'environment': {'ROBOTWIN_ROOT': '/absolute/path/to/RoboTwin',
                            'LINGBOT_ROOT': '/absolute/path/to/lingbot-va'},
            'timeout_seconds': 3600, 'revision': 'filled-by-builder'},
        'operating_point': {
            'name': name, 'tier': 'NUMERIC', 'execution': identity['execution'],
            'remote': {'endpoint': 'ws://model-host:29061', 'identity': identity,
                       'timeout_seconds': 300}}})
Path('/absolute/path/to/arms.json').write_text(json.dumps(arms, indent=2))
```

Choose the allowable success-rate loss and minimum sample count **before** evaluation.
The commands below require `QUALITY_MARGIN` (negative fraction) and `MIN_PAIRS` to be set
explicitly; neither historical −5pp nor any example value is a product default.

```bash
python -m benchmarks.vla.robotwin_driver --make-plan --arms /absolute/path/to/arms.json \
  --profile robotwin_quality_smoke --margin "$QUALITY_MARGIN" --min-pairs "$MIN_PAIRS" \
  --output /absolute/path/to/draft.json
/path/to/simulator/python -m benchmarks.vla.robotwin_driver \
  --prepare-plan /absolute/path/to/draft.json --output /absolute/path/to/scenes.json
python -m benchmarks.vla.robotwin_driver --make-plan --arms /absolute/path/to/arms.json \
  --profile robotwin_quality_smoke --margin "$QUALITY_MARGIN" --min-pairs "$MIN_PAIRS" \
  --scenes /absolute/path/to/scenes.json --output /absolute/path/to/plan.json
python -m benchmarks.vla --registry /absolute/path/to/plan.registry.json \
  run --plan /absolute/path/to/plan.json --output /absolute/path/to/run --gpu 0
python -m benchmarks.vla --registry /absolute/path/to/plan.registry.json \
  report --run /absolute/path/to/run
```

Scene preparation connects to no model: the expert finds valid scenes in a bounded seed
range, avoids duplicate resolved scenes within a task/setting, and freezes the instruction
and initial observations. Both arms must replay those scenes. A draft cannot run as evidence.
Plans, registry snapshots and scene manifests refuse overwrite; use fresh paths when changing
code, configuration or sample sizes. Keep code fixed between planning, serving and running.

`robotwin_quality_smoke` covers one task/seed in each setting and is **always screening,
never reportable**. After wiring and baseline validity are established, use
`robotwin_quality` in both plan commands for 50 tasks × 10 seeds per setting. This sample
count does not guarantee statistical power. Reports apply the explicitly selected budget
using the existing paired Tango method and task-collapse checks; they do not establish
universal generalization or a complete deployment verdict.

## Outputs and extension boundary

Each job emits success, actual executed-action digest, executed steps, frozen scene identity,
server identity and environment provenance. Evidence files contain controller-accepted actions.
Different frozen prompts/observations cannot form a pair even if seed numbers match. Invalid
model output or infrastructure failure currently aborts the job and blocks certification;
this version does not automatically score such errors as task failures. Roundtrip timings
are diagnostic only and include transport; no deadline or end-to-end realtime gate is claimed.

`remote_policy.py` is the reusable bounded transport. `WanVaWireBridge` owns the VA-specific
16×2×16 actions and first-4/subsequent-8 observation history. Other models require their own
observation/action/normalization bridge and a valid native baseline before RoboTwin results
are meaningful. Models incompatible with its embodiment should use a matching simulator
under the same plan/result/report contracts; a universal RoboTwin adapter is not assumed.

Validation so far: protocol and failure-path unit tests, actual local websocket transport
tests, and local RoboTwin expert scene preparation. A pinned-checkpoint full rollout and
baseline A/A campaign remain required before publishing model-quality conclusions.
