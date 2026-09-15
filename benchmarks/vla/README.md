# Reproducible VLA benchmark pipeline

Native simulator setup, paired evidence and supported routes are documented in
[simulator evaluation](SIMULATOR_EVALUATION.md). Policy quality and deadline-aware
evaluation remain separate; the guide records the available routes and their limits.

This directory is the paired accept/reject harness for acceleration and quantization changes.
(The release gates that guard published numbers today are `scripts/check_release.sh` and the
per-family `examples/<family>/reproduce_h100.sh` protocols; this harness takes over
treatment-vs-control decisions once the first real-model paired run has passed through it.) It
answers three different questions without conflating them:

1. **Did serving get faster?** Solo-GPU warm/timed latency samples, control and treatment on the
   same hardware, with deterministic arm-order counterbalancing.
2. **Did the policy's numerical behaviour move?** Seeded contract and held-out open-loop actions,
   checked bit-for-bit or against an explicit model-family envelope.
3. **Did task ability move?** Paired closed-loop LIBERO and RoboTwin episodes, decided by a
   predeclared success-rate margin and matched-pair interval.

A latency win is never accepted as a quality certificate. An open-loop cosine is never presented as
task success. A one-episode smoke is never presented as a release gate.

## Supported matrix

`config/registry.json` is tested against `instinctflash.descriptors.known.KNOWN_DECLARATIONS`:
the built-in ids, backbones, and the V2 numeric envelope are asserted equal by the coherence
test, while the revision pins and determinism classes live only here. It currently covers every
built-in family:

| Family | Default locked checkpoints | Quality surfaces |
|---|---|---|
| GR00T N1.7 | `nvidia/GR00T-N1.7-3B` | contract, latency, DROID-100 open loop |
| LingBot-VA | `robbyant/lingbot-va-posttrain-robotwin` | contract, latency, RoboTwin easy/hard |
| pi0.5 | `lerobot/pi05_base`, `lerobot/pi05_libero_finetuned_v044` | contract, latency, DROID-100, four LIBERO suites |
| DreamZero | `GEAR-Dreams/DreamZero-DROID` | contract, latency, DROID-100 open loop |
| Cosmos3 | Edge and Nano DROID releases | contract, latency, DROID-100 open loop |
| LingBot-VLA | 4B RoboTwin release | contract, latency, RoboTwin easy/hard |
| LingBot-VLA-V2 | 6B RoboTwin release | contract, latency, RoboTwin easy/hard; declared intrinsic numeric envelope |

`SidneyXie/pi05_robotwin` is included as a non-built-in benchmark checkpoint so the supported pi0.5
backbone has a correctly trained RoboTwin closed-loop surface. It does not become a built-in serving
declaration merely by appearing in this benchmark registry.

DROID-only releases do not have a compatible simulator policy head in this repository. For them,
the pipeline honestly reports open-loop regression and latency; it does not invent closed-loop task
success. A fine-tune can be added as a benchmark checkpoint without changing the model family.

## Why environments stay separate

The orchestrator is standard-library Python. Model drivers remain in independent, upstream-compatible
environments. This is a correctness property:

- LingBot-VA's server and RoboTwin's simulator need different Torch stacks.
- GR00T, Cosmos and DreamZero carry different source checkouts and CUDA dependencies.
- Installing one combined environment would silently alter the upstream baseline.

The committed `config/environment.lock.json` defines the boundary. Every driver result must name its
model revision, driver revision and environment fingerprint. The run root also captures the
orchestrator's Python, `pip freeze`, Git status and GPU inventory.

## Quick start

The included CI driver proves orchestration only. It never loads a model and marks its results
`synthetic=true`:

```bash
python -m benchmarks.vla plan \
  --profile smoke \
  --arms benchmarks/vla/config/arms.ci.json \
  --model lerobot/pi05_base \
  --output /tmp/pi05-smoke-plan.json

python -m benchmarks.vla doctor --plan /tmp/pi05-smoke-plan.json
python -m benchmarks.vla run \
  --plan /tmp/pi05-smoke-plan.json \
  --output /tmp/pi05-smoke-run
python -m benchmarks.vla report --run /tmp/pi05-smoke-run --allow-synthetic
```

## The production InstinctFlash driver

`instinctflash_driver.py` is the real-model driver for the built-in families
(pi05, GR00T N1.7, LingBot-VLA-4B, LingBot-VLA-V2). It serves the plan's two preregistered
operating points from one file, one fresh process per job (the V2 lesson: a second model in one
process fails the capture self-check, and you would be timing the loud fallback):

- `stock_upstream` — the family's upstream serving surface, in process, eager: LeRobot's
  processor pipeline + `select_action` (`compile_model=False`, the published rows' eager
  reference), NVIDIA's `Gr00tPolicy.get_action`, or the official LingBot deploy servers'
  in-process `infer` (V2 with `use_compile=False`). These are the `reproduce_h100.sh` stock arms.
- `runtime_default` — `Runtime.from_pretrained` with no flags: what the serve path gives users.

It serves the `contract` and `latency` suites, loads checkpoints only from the local Hub cache
at the plan's locked revision (refusing when `refs/main` has moved), refuses dirty checkouts,
open-loop and closed-loop requests outright, and emits the full result contract with
`synthetic=false`. The `serving` profile plans exactly these two suites at the release
latency protocol (30 timed chunks after 3 warm, repeated contract cases for the
repeatability gate).

```bash
export IFL_BENCH_FAMILY_PYTHON=/path/to/family-env/bin/python  # the family's upstream venv
export IFL_BENCH_IFL_DRIVER=$PWD/benchmarks/vla/instinctflash_driver.py
export IFL_BENCH_DRIVER_REVISION=$(git rev-parse HEAD)

python -m benchmarks.vla plan --profile serving \
  --arms benchmarks/vla/config/arms.instinctflash.json \
  --model robbyant/lingbot-vla-v2-6b-robotwin --output /path/to/plan.json
python -m benchmarks.vla doctor --plan /path/to/plan.json --require-cached
python -m benchmarks.vla run --plan /path/to/plan.json --output /path/to/run --gpu 0
python -m benchmarks.vla report --run /path/to/run
```

Family environment variables (`GR00T_ROOT`, `LINGBOT_VLA_ROOT`, `LINGBOT_VLA_V2_ROOT`) come
from the process environment at run time, exactly as the per-family `reproduce_h100.sh`
protocols document them.

**Validated on real models (2026-08-28, one idle H100 per pair):** the serving profile ran end
to end for four families, every report complete and `reportable=true`, and the numbers agree
with the same-box `reproduce_h100`/verify protocols within ~2% — the two tools validate each
other: LingBot-VLA-V2 661 -> 128 ms (5.19x, numeric envelope), LingBot-VLA-4B 533 -> 165 ms
(3.23x, bitexact), GR00T N1.7 92 -> 51 ms (1.81x, bitexact), pi05 v044 206 -> 91 ms (2.26x,
bitexact — the full `select_action` serve path, which carries ~31-37 ms of processor pipeline
on both arms; the module-protocol chunk pair remains 169 -> 61 ms on this box). The first real
GROOT pair also caught a live product bug: `KNOWN_DECLARATIONS` had drifted from the pointer
package and dropped the fastpath flags, so the bare Hub id served ~11% slower than the row —
fixed and pinned by test. The closed-loop verdict path is validated byte-for-byte against the
frozen `certify()` on the V2 M3 500-pair evidence via `replay_driver.py`. The per-family
`examples/<family>/reproduce_h100.sh` scripts remain the canonical row protocol until all
eight registry families have run through this pipeline (LingBot-VA, DreamZero, and the two
Cosmos3 policies still owe their first pass).

For other drivers, copy `config/arms.template.json`, set the two driver interpreters/scripts,
then materialize only the locked artifacts you need:

```bash
export IFL_BENCH_CONTROL_PYTHON=/path/to/upstream-env/bin/python
export IFL_BENCH_CONTROL_DRIVER=/path/to/control_driver.py
export IFL_BENCH_CONTROL_DRIVER_REVISION=<full-driver-git-revision>
export IFL_BENCH_CANDIDATE_PYTHON=/path/to/candidate-env/bin/python
export IFL_BENCH_CANDIDATE_DRIVER=/path/to/candidate_driver.py
export IFL_BENCH_CANDIDATE_DRIVER_REVISION=<full-driver-git-revision>

python -m benchmarks.vla prefetch --model lerobot/pi05_libero_finetuned_v044
python -m benchmarks.vla prefetch --model lerobot/pi05_libero_finetuned_v044 --execute

python -m benchmarks.vla plan \
  --profile release \
  --arms /path/to/arms.json \
  --model lerobot/pi05_libero_finetuned_v044 \
  --output /path/to/plan.json
python -m benchmarks.vla doctor --plan /path/to/plan.json --require-cached
python -m benchmarks.vla run --plan /path/to/plan.json --output /path/to/run --gpu 0
python -m benchmarks.vla report --run /path/to/run
```

The runner executes argv arrays with `shell=False`, owns an exclusive per-GPU lock, writes requests,
logs and results separately, validates each result before atomic promotion, and resumes only from a
result whose job/request hashes still match. `run_manifest.json` binds the immutable plan to a
credential-free environment manifest; editing either is detected by `report`. Pointing an existing
run directory at another plan is an error.

## Profiles

- `smoke`: one task per suite and very few samples. Wiring only; never reportable evidence.
- `canary`: four tasks, repeated contract cases, 25 DROID observations and two closed-loop seeds.
  Useful for catching obvious regressions before an expensive run.
- `release`: all LIBERO/RoboTwin tasks, 500 DROID observations, repeated deterministic contracts,
  30 latency samples, fifty LIBERO seeds per task and ten RoboTwin seeds per task. This yields 500
  paired episodes per LIBERO suite and 500 per RoboTwin setting/model; increase the profile only by
  committing a new preregistered protocol.

Every seed is serialized in the plan. RoboTwin uses `increment_until_stable`; both arms begin from
the same requested seed, independently skip unstable initializations, and the report rejects the pair
unless the resolved seed is identical.

## Arms and quantized checkpoints

An arm is an execution environment plus an explicit operating point. A quantized arm should override
the checkpoint for each affected model and state the parent revision:

```json
"checkpoint_overrides": {
  "lerobot/pi05_libero_finetuned_v044": {
    "id": "my-org/pi05-v044-fp8",
    "revision": "<full immutable revision>",
    "derived_from_revision": "8e174154ef5f6c60a8da12ae99c303d8963138c1"
  }
}
```

The plan refuses a derived checkpoint whose parent does not match the registry. Multiple treatment
arms may share one control; each receives an independent report and certificate.

## Gates

- **Performance:** candidate/control p50 speedup must meet `min_speedup`.
- **Action:** `registry` means bit-exact for deterministic families and the committed intrinsic
  envelope for LingBot-VLA-V2. A treatment can instead declare an explicit NUMERIC max-absolute and
  cosine gate.
- **Repeatability:** repeated same-arm contract jobs must reproduce the action digest, or stay inside
  the family's declared intrinsic envelope.
- **Closed loop:** paired outcomes use `instinctflash.verify.certify`; LIBERO/RoboTwin preregister a
  `-5pp` margin and Tango one-sided 95% lower score bound. Missing pairs, different resolved seeds,
  task collapse or insufficient sample size are not passes.

The report is `reportable=true` only when every planned result is present, every result validates,
all deciding gates pass, and no synthetic driver participated.

See [DRIVER_CONTRACT.md](DRIVER_CONTRACT.md) to implement a real family driver.

The first remote RoboTwin bridge is available for LingBot-VA. See [ROBOTWIN.md](ROBOTWIN.md)
for frozen scenes, server receipts, explicit quality budgets and screening-only smoke runs.

For the separate official LingBot-VA LIBERO-Long checkpoint and the strict comparison profile,
see [LIBERO_WAN_VA.md](LIBERO_WAN_VA.md).

Model/simulator compatibility is declared in [ADAPTERS.md](ADAPTERS.md).
`python -m benchmarks.vla adapters` lists the explicit contracts; new VA bridge
plans bind checkpoint, observation/action semantics and driver entrypoint before execution.

The installed `instinctflash eval` entrypoint and multi-model simulation workflow
are documented in [Simulator evaluation](SIMULATOR_EVALUATION.md).
