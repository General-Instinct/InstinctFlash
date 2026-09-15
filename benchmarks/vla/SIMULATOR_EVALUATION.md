# Simulator quality evaluation in InstinctFlash

`instinctflash eval` is the installed entrypoint for immutable paired simulation
experiments. Model inference and simulation use separate Python environments and
communicate over identity-checked WebSockets. Simulator dependencies are installed
separately; installing InstinctFlash does not install every model or simulator.

```sh
instinctflash eval adapters
instinctflash eval coverage
instinctflash eval simulator-doctor --simulator robolab
instinctflash eval --registry campaign.registry.json doctor --plan campaign.json
instinctflash eval --registry campaign.registry.json run \
  --plan campaign.json --output campaign-run --gpu 0
instinctflash eval --registry campaign.registry.json report --run campaign-run
instinctflash eval coverage --run campaign-run
```

Coverage distinguishes a proposed simulator route, an implemented contract and
validated results supplied by the caller. A model family appearing in the registry
is not evidence that every checkpoint or robot is compatible.

## Implemented paths

| Model checkpoint family | Simulator | Driver | Protocol |
| --- | --- | --- | --- |
| LingBot-VA LIBERO-Long | LIBERO-10 | `wan_va_libero_driver` | Official VA image/action/history protocol |
| LingBot-VA RoboTwin | RoboTwin | `robotwin_driver` | Official VA client and controller conversion |
| pi05 LeRobot LIBERO | LIBERO | `pi05_libero_driver` | Explicit 10-action execution schedule; frozen actual reset |
| LingBot-VLA 4B | RoboTwin | `joint_robotwin_driver` | 25 × 14 joint/gripper chunk |
| LingBot-VLA V2 | RoboTwin | `joint_robotwin_driver` | 50 × 14 joint/gripper chunk |
| GR00T N1.7 LIBERO fine-tune | LIBERO-10 / Spatial / Object / Goal | `groot_libero_driver` | Native matching fine-tune; 16 × 7 decoded chunk; execute first 8 |

Modules are under `benchmarks.vla`. Exact supported checkpoint revisions,
observation keys and controller semantics are in [ADAPTERS.md](ADAPTERS.md) and
`config/adapters.json`. Checkpoint substitution requires an explicit reviewed
contract. A DROID-trained checkpoint cannot be relabeled as a LIBERO policy.

## Campaign lifecycle

1. Pin the checkpoint, upstream model code, simulator code/assets, task set, seeds,
   numerical settings and candidate passes. Use a custom registry for a new
   simulator-specific fine-tune. Keep clean/randomized as separate suites.
2. Start original and candidate endpoints in their model environments. The
   `joint_policy_server` supports the two joint-action LingBot checkpoints;
   `groot_policy_server` supports a local view of the pinned LIBERO fine-tune.
   Each endpoint loads the real policy and checks output geometry before emitting
   its identity receipt. A shape probe is not quality evidence.
3. Build a draft plan with the explicit driver/adapter. Prepare scenes using
   `python -m benchmarks.vla.<driver> --prepare-plan draft.json --output scenes.json`.
   Set the simulator-specific source environment variables documented below.
4. Bind the receipt under each arm's `operating_point.remote.identity`, its
   WebSocket `endpoint`, and `execution`. Bind the frozen scene file's absolute
   path and SHA-256 under `operating_point.scene_manifest`. Build a fresh final
   plan. Match weight/source/package/numerical identities across paired arms;
   record intentional numeric or behavioral differences explicitly.
5. Run and report through the standard CLI. Do not edit the frozen pipeline,
   checkpoint, adapter or scenes during a campaign. Resume with the same plan.
   Start a new campaign if any of those inputs changes.

The joint driver needs `ROBOTWIN_ROOT` and `LINGBOT_ROOT` (the pinned VA upstream
client supplies environment construction/expert scene preparation only). Model
servers need `LINGBOT_VLA_ROOT` or `LINGBOT_VLA_V2_ROOT`. The GR00T driver needs
`GR00T_ROOT` and `LIBERO_ROOT`; its simulator environment must import the upstream
GR00T `LiberoEnv` and its pinned LIBERO installation.

GR00T uses the official native seeded reset, two-axis RGB flip, EEF state and
upstream gripper conversion. It adds no settling steps or init-state replacement.
The actual initial observation must match its frozen digest. Controller traces
are recorded after upstream conversion. The native policy's model horizon (40)
and LIBERO decoder horizon (16) are distinct. Its checkpoint view contains the
fine-tune's JSON/weight files and an `instinctflash.json` declaration with backbone
`groot_n17`, embodiment `libero_sim`, action NFE 4 and a local `base_weights` path.
The server verifies the view against the selected fine-tune subdirectory in the
pinned official snapshot. Implemented suites and completed evaluations are
reported separately; supporting a contract does not mean its experiment has run.

On this H100 host, LIBERO uses `MUJOCO_GL=osmesa`,
`PYOPENGL_PLATFORM=osmesa`, `LP_NUM_THREADS=1` and fixed process seeds. Freeze and
verify renderer outputs on your own host; another renderer is a new protocol
configuration, not an interchangeable pixel source.

## Separate the conclusions

- **Observed action equivalence:** compare the complete executed controller trace
  on matching frozen scenes/noise. A matching trace does not prove hidden-state or
  universal bitexact equivalence.
- **Quality:** compare paired success rates, retain failures, report uncertainty
  and sample size. Smoke/screening runs never become release certificates merely
  because every observed episode succeeds.
- **Speed and realtime:** measure separately under the target device and control
  budget. Paused simulator roundtrip timings are diagnostics, not a Thor realtime
  result. Rendering and inference may share a GPU during quality screening.
- **Optimization tier:** bitexact candidates, numerical changes and behavioral
  changes such as distillation/NFE reduction must be separate named arms. A success
  tie cannot turn a numeric action mismatch into a bitexact result.

The present pi05 protocol retains NFE as the arm setting, 50-action policy chunks,
10 executed actions, TF32 enabled, cuDNN benchmark enabled, compilation disabled.
It is not the checkpoint's default 50-action execution schedule. Original-repeat evidence is not an original-versus-optimized comparison;
the separate capture protocol below supplies the latter when executed.

## Remaining external routes

Cosmos Edge/Nano use official RoboLab DROID clients; DreamZero uses its DROID
`sim-evals` client. These are **routes, not implemented evaluated adapters** in
this component. Their Isaac Sim renderer requires a supported RTX GPU. NVIDIA
explicitly excludes H100/A100 from supported rendering hardware:
[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html).

`simulator-doctor` detects the known H100-only local-renderer blocker. It does not
certify an unfamiliar GPU or verify simulator installation/assets. Deploying the
simulator on a supported RTX host and retaining inference on H100 is a viable
layout. Freeze that simulator's assets, native observation/action protocol and
reset evidence before advertising quality coverage. No dummy images, projected
robot actions or synthetic successes substitute for that evaluation.

## Repeatable screening without host-specific planning scripts

`config/campaign.groot_libero.json`, `campaign.vla4_robotwin.json`,
`campaign.vla2_robotwin.json` and `campaign.pi05_libero_capture.json` are explicit
10-task, two-seed templates. Copy the appropriate template and replace its
`/path/to/...` interpreter, source checkout and endpoint receipt paths. Start the
model endpoints described above for GR00T and the joint-action LingBot models.
The pi05 driver starts its own separate original/capture episode servers.
Install the bundled `examples/pi05_vla` package into that driver environment
(`pip install --no-deps -e /path/to/frozen/execution/examples/pi05_vla`);
`doctor` checks its adapter entry point before a capture campaign.

```sh
instinctflash eval sim-plan --spec campaign.json --output draft.json
instinctflash eval prepare-scenes --plan draft.json --output scenes.json \
  --repo-root /path/to/frozen/execution --workers 2 --gpu 0 --gpu 1
instinctflash eval sim-plan --spec campaign.json --scenes scenes.json --output plan.json
instinctflash eval --registry plan.registry.json doctor --plan plan.json
instinctflash eval --registry plan.registry.json run \
  --plan plan.json --output run --repo-root /path/to/frozen/execution --gpu 0
instinctflash eval --registry plan.registry.json report --run run
instinctflash eval --registry plan.registry.json export --run run --output evidence
instinctflash eval verify-bundle --bundle evidence
```

For CPU OSMesa LIBERO rendering, omit preparation `--gpu` flags. `--gpu` on `run`
selects the driver process GPU; remote policy endpoints keep their own device
assignment. Use independent model and simulator environments when dependencies
conflict. Do not resolve a virtualenv's `bin/python` symlink to the base interpreter.

Scene preparation isolates each task and setting in a fresh process, preserving
all arms and seeds for that task. This avoids accumulating native renderer
contexts across tasks. Completed shards have hashes and completion receipts;
rerunning the same command resumes verified shards. Merge refuses missing scenes,
duplicate keys or changed source/assets. A failed preparation is an infrastructure
error, not a policy failure. The native expert gate retries unstable scenes and
its specific unreachable-grasp assertion; unrelated assertions and renderer errors
abort preparation. To reuse verified whole-task manifests from a different frozen
preparation attempt, pass repeated `--reuse-scenes path/to/shard.scenes.json`. The
merged artifact records input hashes and the old/new preparation identities;
partial seed coverage and duplicate scenes are refused. Never replace a failed task with a different task to
improve the score.

The builder pins simulator git revisions, explicit checkpoint contracts and paired
endpoint identities. It refuses unsupported checkpoints, silent task truncation,
overwriting existing plans, and labeling the default numeric V2 Runtime as bitexact.
All generated plans are screening-only. Their five-percentage-point gate field is
a template placeholder, not an agreed acceptable quality loss; it cannot authorize
a release. An arbitrary new checkpoint still needs a
reviewed observation/action/reset contract; a shared model family is insufficient.

pi05's optional `operating_point.optimization="instinctflash_capture"` preserves
NFE=10 and installs the real adapter's loop-constant hoists and static-KV capture
on the same verified LeRobot evaluation policy. Its original and capture variants
have different sockets and model objects, preventing mutation of the original.
Both retain the legacy 10-action schedule, identical processors and numerical flags.
The result records installed passes and live self-check evidence, including fallback.
This isolates adapter acceleration from the full Runtime facade and its FP32
precision lease; it must not be described as a full default Runtime comparison.
The capture adapter supports both legacy two-tensor and current three-entry
Transformers cache iteration. Legacy direct cache indexing returns only the live
prefix; this prevents an eager suffix concatenation from reading the reserved
static suffix twice. Both cache layouts have CPU regression checks, and the live
replay self-check remains mandatory.

Live pi05 plans still require a committed execution checkout. Create and commit an
isolated snapshot before binding the final plan; do not commit unrelated workspace
changes just to satisfy this requirement.

Reports include `paired_success` even below the release gate's minimum sample size:
control/candidate rates, candidate-minus-control delta, discordant pairs, per-task
counts and a Tango **central 95%** score interval. This is descriptive, assumes
matched Bernoulli episode pairs and does not model task clustering. It is distinct
from the release gate's one-sided 95% bound / central 90% pair. Matching all observed
successes does not collapse the reported uncertainty to zero. Neither this table
nor successful action-byte comparison overrides a failed/incomplete release gate.

Exported evidence contains original plans, reports, results/action traces, logs,
registry and frozen scene files with a file inventory. Original absolute paths are
preserved in the immutable plan; a mapping locates bundled scene copies. The bundle
can be checked and analyzed on another host without the live model/simulator.
Re-execution requires restoring dependencies and generating a fresh plan for the
new paths. File hashes detect corruption; they are not a cryptographic signature
from a trusted third party.

For scale, 10 tasks × two seeds gives 20 paired episodes per setting. If every pair
has the same success outcome, the central 95% paired score interval is still about
±16.1 percentage points. This is useful regression screening, but cannot establish
a five-point quality-loss budget. Increasing seeds improves episode-level precision;
covering more tasks addresses task coverage, which that interval does not measure.

RoboTwin scene manifests require `reset_policy="fresh-clutter-bounds-v1"`.
The upstream clutter sampler mutates default `xlim`/`ylim` lists when applying a
table offset. Repeated resets can therefore change distractors despite an identical
seed. The driver copies bounds for each call during both preparation and rollout,
without modifying the simulator checkout. Old scene manifests must be regenerated;
an initial-observation mismatch aborts the episode as an infrastructure error and
never becomes a policy failure or a skipped task.

For two-arm closed-loop campaigns with independent remote policy servers, use
`run --paired-workers 2` to run both arms of one pair concurrently. Each endpoint
serves at most one episode at a time; the next pair waits for both current episodes.
This mode refuses latency plans, noncontiguous pairs and shared endpoint URLs.
Use genuinely separate server instances, not URL aliases for the same server.
The scheduling choice is recorded in the run environment and cannot change on
resume. Fail-fast lets already-started paired episodes finish before stopping.
Concurrent rendering/inference timings are unsuitable for latency claims.

## Attribute action differences before changing model quality

Joint RoboTwin and GR00T campaigns accept `comparison: "aa"`, `"bb"` or `"ab"`.
A means the original endpoint, B the Runtime endpoint. Bind two A receipts for
A/A, two B receipts for B/B, and A/B for the normal comparison. Use the same
frozen scenes and **serial** scheduling for all three experiments. Shared endpoint
A/A or B/B cannot use `--paired-workers 2`. The pi05 capture template currently
supports A/B only. Tier declarations are hypotheses to check; failed original
repeatability must remain visible in the report.

For native V2, start both `joint_policy_server` instances with `--capture-noise`
and set `record_inputs: true` in the campaign. Each result's adjacent trace records
the real wire observations, reset seed, actual sampler noise tensors and responses,
with a file hash inventory. The noise hook preserves the native draw's shape,
dtype, device and RNG position. A trace marked complete means its wire recording
was closed; successful episode completion is established by the result artifact.
These recordings can be large because they include camera images.

```sh
instinctflash eval replay-policy --trace /path/to/control.pending.trace \
  --endpoint ws://127.0.0.1:29571 --receipt stock.receipt.json \
  --repeats 3 --output stock.fixed-input.json
instinctflash eval replay-policy --trace /path/to/control.pending.trace \
  --endpoint ws://127.0.0.1:29572 --receipt runtime.receipt.json \
  --repeats 3 --output runtime.fixed-input.json
```

Repeat `--trace` for multiple recordings. Replay requires matching checkpoint and
protocol, seeds each recorded episode, and injects the **stored initial tensor**;
reusing a seed alone is insufficient. It reports noise identity, within-endpoint
repeatability and differences from recorded actions. This isolates feedback from
the simulator; it does not replace fresh closed-loop success evaluation. Run it
in an environment with the `eval` and `serve` extras installed.

## Measure the actual edge device and declare its control budget

`measure-policy` currently implements native V2 on the local CUDA device. It
verifies the checkpoint inventory and native source against a real recorded trace,
keeps NFE=10, and runs the Runtime arm in-process. Original and Runtime must run
in separate fresh processes. The probe stores raw samples, device/packages,
numerical flags, source hashes and Runtime explanation. Warmup and startup are
separate from measured calls. `--tier-ceiling numeric` explicitly allows V2 native
capture; the default BITEXACT ceiling declines its numerical-envelope gate. The
probe always uses `in_process`, so this option does not select the FP8 engine.

```sh
instinctflash eval measure-policy --trace real-input.trace \
  --model robbyant/lingbot-vla-v2-6b-robotwin --revision PINNED_REVISION \
  --mode stock --warmup 8 --iterations 128 --output stock.latency.json
instinctflash eval realtime-report --measurement stock.latency.json \
  --budget control-budget.json --output stock.realtime.json
```

An explicit example budget is:

```json
{"target_device":"Thor","control_hz":50,"executed_actions":50,
 "scheduling":"pipelined","max_deadline_miss_fraction":0}
```

This example allows 1000 ms to replenish a 50-action chunk **only when execution
and inference overlap**. A blocking controller gets one control period (20 ms at
50 Hz). Set `max_observation_to_action_ms` to impose a tighter reaction deadline.
Chunk throughput is not observation-to-action responsiveness. The probe measures
CPU preprocessing through synchronized ready CPU actions, including diagnostic
noise-copy overhead; camera, network, actuator and sustained controller scheduling
are excluded. A passing observed sample budget is not deployment certification.
If the original meets the actual budget, preserve its quality; speed alone never
authorizes a NUMERIC or BEHAVIORAL arm.

## Onboard another native checkpoint through the same CLI

The reviewed GR00T contract supports `libero_10`, `libero_spatial`, `libero_object`
and `libero_goal` subdirectories of the pinned `nvidia/GR00T-N1.7-LIBERO` revision.
These are distinct fine-tuned weights, with suites bound to their native semantics.
Create a verified view without copying large weights:

```sh
instinctflash eval checkpoint-view --model nvidia/GR00T-N1.7-LIBERO \
  --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
  --subdir libero_spatial --output /path/to/spatial-checkpoint
```

Start `groot_policy_server` with that view and
`--checkpoint-subdir libero_spatial` in each model process. Add
`"checkpoint_subdir": "libero_spatial"` to the standard GR00T campaign template;
its matching suite is selected automatically. Follow the same plan, prepare,
doctor, run, report, export and verify lifecycle above. Wrong-suite substitution
and mismatched endpoint weights are refused. This is reusable within a reviewed
native contract; arbitrary embodiment/action-space conversion still needs an
adapter, not just a checkpoint path.

Before retrying an interrupted or failed job, the runner archives its prior request,
log, failure record, pending output and any recorded wire trace under `failures/attempts/<job_id>/` with
content hashes. Export includes those records. A retried infrastructure failure is
not silently converted into a policy failure or erased from the evidence history.


## Bind quality and latency to an execution

Use `measure-endpoint` to time the same pinned endpoint used by a quality campaign,
then `execution-report` to attach its exact profile to verified evidence bundles.
`select-configuration` evaluates those records against explicit control, precision,
step and task-quality constraints. It retains a sufficient native baseline and
refuses to promote screening evidence into a deployment certificate.
The [CLI](cli.py), [execution evidence](execution_evidence.py) and
[configuration selection](configuration_select.py) define the commands, required
identity fields, policy schema and exit statuses. H100 quality and Thor timings
cannot be merged into a single certified profile.
