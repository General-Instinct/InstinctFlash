# Native defaults and checkpoint qualification

The default path preserves native schedules and precision. Optional V2 NUMERIC
capture remains unsuitable for automatic selection: admission is intermittent,
and the legacy velocity guard has no same-domain calibration. No new closed-loop
accuracy certificate is claimed here.

This campaign started September 6 and finished September 7, 2026. It adds fresh
startup checks, explicit failure retention and target-device evidence to the
[existing simulator results](../simulator_quality_2026-09-06/README.md).

## Repairs

- Capture checks snapshot the eager reference before replay can overwrite reused
  storage. V2 checks each staged velocity input three times without increasing
  the old threshold. π0.5's private check also snapshots its reference.
- V2 GPU preprocessing obeys the NUMERIC plan. Rejection removes preprocessing
  and all prefix/denoise graphs, discards the in-flight mixed call, restores RNG
  and server counters, and reruns upstream. Experimental custom kernels refuse
  further serving when that complete fallback cannot be guaranteed.
- LingBot-VA Hub declarations now preserve RoboTwin **25V/50A** and LIBERO
  **20V/50A**. The old implicit RoboTwin 2V/4A declaration was caught by the new
  native-schedule assertion. Reduced schedules remain explicit choices; existing
  user-authored checkpoint declarations are not rewritten.
- VA's native config determines temporal history: **4/8 frames** for RoboTwin,
  **12/16 frames** for LIBERO. The public observation contract and deferred KV
  commits agree. Worker placement also commits the executed action with the next
  observed frame window. Its launcher verifies the declared temporal chunk.
- Native frozen components resolve from the pinned package, rather than a second
  lookup of `main`. The worker binds that path into the selected native config,
  includes its entrypoint in the wheel, and drains verbose child output so a full
  pipe cannot stall loading or prediction.

## Observed startup admission

There are **68 final startup receipts: 62 ordinary attempts and six fault drills**.
Counts were declared before outcomes. A rejected attempt is never replaced by a
successful retry. Three successful starts are an engineering check, not a
production reliability bound or task-success estimate.

| Runtime checkpoint/path | Device | Admitted / ordinary attempts |
| --- | --- | ---: |
| LingBot-VLA 4B, native capture | H100 | 4/4 |
| π0.5 base, native capture | H100 | 3/3 |
| π0.5 LIBERO, native capture | H100 | 3/3 |
| GR00T LIBERO-10 / spatial / object / goal | H100 | 3/3 each |
| LingBot-VA RoboTwin, native schedule | H100 | 3/3 |
| LingBot-VA LIBERO, native schedule | H100 | 3/3 |
| V2 default BITEXACT path | H100 / Thor | 3/3 each |
| V2 optional NUMERIC capture | H100 | 2/8 |
| V2 optional NUMERIC capture | Thor | 0/4 startup trials; 1/1 separately predeclared serving attempt |

All six fault drills were rejected: V2 on both devices, VLA 4B, GR00T LIBERO-10,
and both π0.5 checkpoints. Actual rejected V2 receipts show preprocessing,
vision, prefill and denoise capture inactive. Ordinary reference starts are also
retained in [analysis.json](analysis.json).

VA receipts observe eight predictions and seven native deferred commits, rather
than pretending that eight inference-only calls exercise this model. All three
Runtime processes per VA checkpoint reproduce the stock process's eight
action-value digests on these seeded synthetic inputs. This is bounded API/state
progression evidence, not byte certification over arbitrary inputs, ring
saturation, or a simulator success rate.

The final [real LIBERO worker smoke](worker-smoke.json) also reproduces all eight
stock action-value digests through `Runtime(..., placement="worker")`. Three
earlier attempts failed before prediction and led to concrete fixes: resolving
`main` instead of the pinned frozen stack, the native config's placeholder weight
path, and an upstream client loaded with an invalid relative import. They are
preserved as setup failures; the successful smoke is separate from the 68
startup receipts. The managed client now uses the shipped wire codec.

Ten initial π0.5 attempts used an incompatible LeRobot 0.4.4 environment and
failed before inference. The fixed protocol was rerun once in the existing
LeRobot 0.6.1 environment. Seven preliminary VA attempts exposed an incorrect
probe expectation for the 16D RoboTwin wire, the implicit reduced schedule, or
the missing LIBERO declaration. Their logs, amendments and frozen source are
retained separately from final admission counts.

## Thor latency and explicit budgets

Each endpoint used eight warmup calls and 128 measured calls on real recorded
observations. These are localhost websocket roundtrips, including serialization
and transport. Recorded noise tensors were not injected. Initial endpoints were
measured separately and released before measuring the strict endpoint.

| V2 execution | p50 | p99 |
| --- | ---: | ---: |
| Original | 717.29 ms | 764.51 ms |
| Default BITEXACT Runtime | 717.61 ms | 778.86 ms |
| Admitted NUMERIC capture | 406.60 ms | 409.67 ms |

For the illustrative 50-action, 20 Hz pipelined budget, the 2500 ms replenishment
deadline keeps the native reference. Adding a 500 ms reaction deadline makes
capture fast enough in this observed run, but selection refuses it for lack of
a matching paired quality certificate. A 50 Hz blocking budget is missed by all
three. These examples are not application-specific deployment approval. See
[the exact selection outputs](thor-budget-selections.json).

## V2 repeatability diagnosis

The old `0.05083918571472168` threshold came from **controller actions** in
`moe_kernel_results.json`; startup compares **denoise velocity**. It is now
labelled `legacy_cross_domain_guard`, not a calibrated velocity envelope.

On the earlier frozen source, fixed-input velocity diagnostics observed H100
A/A, A/B and B/B maxima of 0.0859375. Thor's captured-chunk B/B reached
0.05419921875 even though its first A/B comparison could be small. This localizes
variation before simulator feedback; it does not prove a particular kernel is
the sole cause, or that the graph adds no error.

Separately, twelve real observations with their actual recorded noise were
replayed three times per process. Across reference/reference,
reference/admitted-capture and admitted-capture/admitted-capture comparisons,
the largest action delta was 0.011467278003692627 on both devices. Equality of
these observed maxima is not a non-inferiority certificate. Instrumented
diagnostics, normal admission attempts, fault drills and serialization pilots
are kept separate.

## Shared workflow and remaining scope

Use [`startup-probe`, `execution-report`, `qualification-report`, and explicit
budget selection](../../docs/rfc/execution-evidence.md). The inventory covers
every registered checkpoint and simulator contract, with **26 distinct observed
profiles** and **12 verified historical bundles** in this campaign. It never
transfers quality across source versions, checkpoints, GR00T variants or devices.

[H100 inventory](h100-qualification.json) and [Thor inventory](thor-qualification.json)
include missing work as well as measured paths. An older GR00T bundle without a
checkpoint subdirectory is retained as unassigned historical evidence. The
GR00T DROID base, π0.5 base/RoboTwin, Cosmos Edge/Nano and DreamZero do not inherit
the simulator qualification of their fine-tunes or other families. Missing
native contracts remain explicit; Cosmos/DreamZero closed-loop evaluation also
still needs an appropriate supported rendering environment.

All execution records remain `deployment_certified: false`. No FP8, distillation,
automatic extra-seed campaign, or unrequested 50-task rollout was introduced.

## Reproduction

Raw root: `/home/ubuntu/ifl_eval/native_qualification_20260906`.
Frozen `execution` is the attribution phase; `final-execution` holds capture
repairs; `va-final-execution` holds native VA declaration/history repairs.
Worker repair attempts have separate frozen checkouts. Later changes do not
retroactively update the identity of an earlier GPU run.

The final worker checkout's Runtime and benchmark Python sources match the
finished workspace. Forty-five targeted regression tests passed, together with
the existing Runtime facade, precision, geometry, scaffold and serving-parity
checks. Startup checks and action-value digests do not replace simulator quality
evaluation.

```sh
python eval/native_qualification_2026-09-06/analyze.py \
  --root /path/to/restored-artifacts --output /new/path/analysis.json
python eval/native_qualification_2026-09-06/build_inventory.py \
  --root /path/to/restored-artifacts --repository /path/to/InstinctFlash \
  --output /new/path/inventory
```

The inventory builder also needs the twelve prior bundles at their declared
locations. Execution records must be rebuilt after relocating their source
files. Original launch commands, environment amendments, real input traces and
rejected receipts are preserved; model weights are referenced by pinned
revision and inventory hash rather than copied into the evidence archive.

The raw archive is 12,070,319 bytes; all **2,971 evidence/source files** were
read back and verified. SHA-256:
`147d73a18a7683d8b902595d4f197e3ae7110f0ddef8179e4422f9e618f9a959`.
See [artifact-index.json](artifact-index.json) for paths, wheel hashes and
verification metadata. The installed wheel includes the worker entrypoint and
reproduces the qualification report outside the repository, without a model
stack. The final worker's Runtime/benchmark Python sources match the finished
workspace. All owned H100/Thor model processes were closed; H100 GPUs 4–7 were
at 0 MiB afterward.
