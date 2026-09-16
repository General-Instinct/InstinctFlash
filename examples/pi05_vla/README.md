# pi05, a VLA in InstinctFlash

`lerobot/pi05_base` is a vision-language-action policy, registered from outside the core through the
`instinctflash.adapters` entry point. It is here because it is structurally unlike LingBot-VA in almost
every way the runtime cares about, which makes it a test of whether InstinctFlash's declarations describe
*execution* or merely describe one world model.

| | LingBot-VA (world-action) | pi05 (VLA) |
|:--|:--|:--|
| streams | two coupled: video + action | one: a prefix |
| observation history | growing ring, 72-frame window | `n_obs_steps=1`, a single observation |
| K/V across control steps | carried and **grown** | prefix, **recomputed every step** |
| K/V lifetime | `EPISODE` | `CHUNK` |
| guidance | CFG at 5.0 on video | none — flow matching |
| forwards per control step | checkpoint-specific video/action schedule | 11 (1 prefix + 10 flow steps) |
| action chunk | 32 | 50 |
| commit phase | yes, a deferred ring advance | none |
| language | a prompt, encoded once per episode | a prompt, **tokenized by a processor pipeline** |

Every fact in the right-hand column is read from that checkpoint's own `config.json`, not guessed:
three cameras at `(3,224,224)`, a 32-dimensional state, `num_inference_steps: 10`, `chunk_size: 50`.

## What pi05 needs that a world model does not

**A processor pipeline, and it is not optional.** `predict_action_chunk` reads
`batch[OBS_LANGUAGE_TOKENS]` and `batch[OBS_LANGUAGE_ATTENTION_MASK]` — already tokenized. Text never
reaches the policy. The tokenizer, the input normalisation and the action un-normalisation all live in
a `PolicyProcessorPipeline` published beside the weights as `policy_preprocessor.json`. A VLA served
without it is not slow, it is **wrong**: fed unnormalised pixels, returning actions in a normalised
space nobody can execute. `build_in_process` therefore loads the policy *and* its pipeline, and maps
the declaration's `prompt` onto LeRobot's `task` key — model semantics, so it stays in the adapter.

**A patched `transformers`.** pi05 asserts
`transformers.models.siglip.check.check_whether_transformers_replace_is_installed_correctly()`, which
standard transformers does not provide. Upstream ships the replacement files in openpi's
`transformers_replace/`, and newer LeRobot exposes them as `pip install "lerobot[pi]"` — an extra that
`lerobot 0.4.4` does not have. Until that environment exists, `build_in_process` raises with the real
reason instead of a `ValueError` about a version.

## The concept this comparison contributed

Three families made one property visible that none of them declares directly: whether tensor shapes
repeat from one control cycle to the next. A stream that outlives a cycle accumulates, so the extent
read on cycle N differs from cycle N-1 and a captured graph is invalid. That is derivable from the
declared stream lifetimes, so `AdapterSpec.shapes_static_across_cycles()` derives it:

    LingBot-VA   GROWS    streams ['action', 'video'] outlive a control cycle
    pi05         STATIC   all streams (prefix) are rebuilt within a control cycle

Whole-cycle graph capture measured **1.43x slower** on LingBot-VA and is the headline optimization of
hand-tuned VLA engines. Both are consequences of that one line.

## Graph capture is the default, and the self-check is the reason it can be

On capture-capable devices (CUDA with graph support; the measured bandwidth-bound-edge class
declines at plan time with its 1.04x reason), every pi05-class checkpoint — `pi05_base`, published
fine-tunes, and a directory fresh out of `lerobot-train` — serves on the replay-safe **static-KV
CUDA-graph capture** by default (`pi05_iwm/static_capture.py`; measured 206.7 → 72.8 ms/chunk on
H100/v044). No flag, no `compile_model` trigger. It used to be opt-in because capture was gated on
evidence measured on *other* checkpoints; what makes the default safe is that the evidence is now
re-earned **per process**:

- **The bit-exact self-check.** Immediately after the first capture, replay is compared against
  upstream eager `denoise_step` on staged inputs the capture never saw — fresh `x_t` draws from a
  dedicated generator, every warmed schedule timestep, and a synthetically *refilled* prefix so a
  graph that baked K/V values instead of reading the live buffers cannot pass. Exact equality
  (`atol=0`). PASS → replay serves, and the plan's `graph_capture` entry gains the line
  `self-check bit-exact on N inputs`. FAIL → the graphs are released, `denoise_step` is rebound to
  upstream, the observed delta is printed and recorded on the plan, and serving continues on eager
  arithmetic. The check costs seconds, once per process, at first capture.
- **Kill-switch:** `IFL_PI05_NO_CAPTURE=1` serves eager (recorded on the plan, printed). It is
  refused on checkpoints that *declare* the TF32 static-KV operating point, because there eager
  would be a different execution semantics than the declared one.
- **Two-graph path:** H100 and Thor default to prefix + full-loop graphs with fixed-step tables
  after matched native qualification (H100 97 → 86 ms; Thor 324 → 309 ms). `IFL_PI05_FULL_CHUNK_GRAPH=0`
  restores the per-step graph; `IFL_PI05_PREFIX_GRAPH=0` disables only the prefix graph.
  Other devices can explicitly select `IFL_PI05_PREFIX_GRAPH=1` to capture both the original
  vision/language prefix and fixed Euler loop. Dynamic NFE and RTC keep the default per-step path. This remains BITEXACT and
  gets a whole-`sample_actions` self-check on fresh noise and changed prompt/mask/image bytes. On
  RTX 5090 with `pi05_libero_finetuned_v044`, 21-run medians were 130.0 → 89.4 ms/chunk (1.45x),
  with 0.091 GiB additional allocated memory and 1.38 s one-time self-check cost. Adding
  `IFL_PI05_FULL_STEP_TABLES=1` computes the fixed schedule's exact time-MLP/AdaRMS outputs once,
  binds each unrolled step to its own static tensor addresses during capture, then restores the
  real modules. It measured 128.7 → 85.7 ms (1.50x; another 3.7 ms), remained bitexact after
  switching to dynamic NFE, used 0.098 GiB additional allocated memory, and self-checked in 1.36 s.
- **Retired opt-ins:** `IFL_PI05_STATIC_CAPTURE=1` (now the default) and `IFL_PI05_CAPTURE=1`
  (the DynamicCache experiment, measured replay-unsafe) are no-ops with a notice.

## RTX 5090 FP8 qualification

The FlashRT SM120 path is now qualified against the real
`lerobot/pi05_libero_finetuned_v044` checkpoint at revision
`8e174154ef5f6c60a8da12ae99c303d8963138c1`, using real observations from
`lerobot/libero_spatial_image` revision `d86c0b94922572b3b657e1d1a3d01f0952ddeb46`.
Native BF16 and FP8 run in separate processes with identical prompt, observation rows, and
bitwise-equal diffusion-noise tensors. SM120 must select the transpose-B `nk` layout; CUDA 12.8
cuBLASLt rejects the previous `kn` descriptor for production Pi0.5 shapes.

The current gate runs through public `flash_rt.load_model`, explicit multi-frame
`calibrate(..., prompt=...)`, and `predict(..., state=...)`. The checkpoint's own
LeRobot processors normalize state, encode its 32 padded state entries into the
prompt, resize images, and decode actions using the declared **MEAN_STD** statistics.
The engine computes the checkpoint-native **50** action steps and returns **10**
for LIBERO replanning. Missing state and unsupported public horizons are refused.
Both FP8 steady residency and startup peak are gated at at most 0.75 of native
PyTorch allocated memory. Full current measurements are in `sm120_fp8_results.json`.

Current public-API replay medians are **41.57 ms BF16 → 24.60 ms FP8 (1.69x)**.
FP8 peak allocated memory is **4.01 GiB**, with **3.89 GiB** steady allocation
versus **6.37 GiB** native. These are PyTorch allocator measurements, not total
process VRAM including raw CUDA/cuBLAS/EGL allocations. The three numeric cases
use four calibration frames at percentile 99.9; the task campaign below uses
its separately frozen eight-frame, percentile-99.0 calibration.

Earlier 1.546x / 9.15 GiB measurements used the legacy FlashRT contract: no state
tokens, a computed chunk of 10, and quantile action decoding. They are not a
checkpoint-semantic qualification and must not be compared as the same operating
point. The new streaming quantizer avoids the full BF16 GPU weight staging that
caused that startup peak.

The numerical gate also exercises 12 token lengths against an eight-entry graph
cache and verifies eviction/close destroys the graph handles. This is explicitly
shape-only stress and is excluded from task-quality and timing measurements.
Decoder attention-output autotune now allocates by the actual `M*K` read size:
`K=2048` does not fit its old `DEC_D=1024` scratch choice. CPU capacity tests
cover both single/batched and 10/50-step cases; the GPU regression was checked
with Compute Sanitizer.

The separate LeRobot Runtime adapter's 50-action *execution queue* is exercised by
`run_pi05_end_to_end.py`. Reproduce the corrected SM120 gate with:

```bash
hf download lerobot/pi05_libero_finetuned_v044 \
  --revision 8e174154ef5f6c60a8da12ae99c303d8963138c1 \
  --local-dir /path/to/pi05_libero
hf download lerobot/libero_spatial_image --type dataset \
  --revision d86c0b94922572b3b657e1d1a3d01f0952ddeb46 \
  --include data/chunk-000/file-000.parquet --local-dir /path/to/libero_spatial

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=serving:. python \
  examples/pi05_vla/verify_sm120_fp8.py \
  --checkpoint /path/to/pi05_libero \
  --dataset /path/to/libero_spatial/data/chunk-000/file-000.parquet \
  --output examples/pi05_vla/sm120_fp8_results.json
```

The complete protocol, source hashes, per-arm memory/timing, and per-case comparisons are in
`sm120_fp8_results.json`.

### RTX 5090 LIBERO closed-loop screen

The historical legacy FlashRT binary and checkpoint completed a `libero_spatial` screen using
`hf-libero==0.1.4`, robosuite 1.4.0, MuJoCo 3.8.1, and the immutable
`lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` assets. Ten tasks × three
episodes produced **21/30 successes (70.0%)**. Per-task successes were
`[2, 1, 3, 3, 2, 2, 3, 0, 3, 2] / 3`; every task loaded, rendered non-blank EGL observations,
calibrated, captured, and completed without a runtime failure.

This is deliberately labelled a **SCREEN**: three episodes per task are not a statistical
non-inferiority certificate, and this run did not include a matched native-precision arm. The
complete protocol, package/source/binary hashes, task descriptions, timings, and limitations are
in `sm120_libero_screen_results.json`. Its source hashes remain unchanged. It also
omitted state tokens and used a different action decoder; this record is retained
for historical audit only. Reproduction requires the original source revision
`08771e1`, not the current API, which deliberately refuses missing checkpoint state.

## Matched RTX 5090 LIBERO qualification

`benchmarks/vla/pi05_libero_driver.py --matched-config CONFIG.json` now has a
FlashRT Native/FP8 campaign. Native means the FlashRT BF16 implementation here;
this does not certify the separate LeRobot Runtime 50-action queue. Both arms use
the checkpoint-native computed chunk of 50, 10 actions per replan, NFE=10,
10 settling steps and the LeRobot Spatial 280-step episode limit.
Seeds 40100–40149 select all 50 initial states, with Python, NumPy and Torch
reseeded per episode. A single synchronous environment runs at a time. The
second arm consumes a checked reset against the first arm's recorded state and
camera digest; policy noise hashes must match for every common inference call.

### Completed 500-pair result

The full Spatial campaign passed the pooled non-inferiority gate:
**Native 472/500 (94.4%), FP8 479/500 (95.8%)**. The difference is **+1.4 percentage
points**, with a Tango paired central 95% interval of **[-0.135, +3.198] points**.
The one-sided 95% lower bound is **+0.138 points**, above the preregistered
**-5-point** margin. No task collapsed. This establishes non-inferiority; the
central 95% interval crosses zero, so no superiority claim is made.

| Spatial task | Native / 50 | FP8 / 50 |
| --- | ---: | ---: |
| 0 | 50 | 50 |
| 1 | 50 | 48 |
| 2 | 50 | 50 |
| 3 | 50 | 50 |
| 4 | 47 | 48 |
| 5 | 28 | 33 |
| 6 | 50 | 50 |
| 7 | 49 | 50 |
| 8 | 49 | 50 |
| 9 | 49 | 50 |

`sm120_libero_matched_results.json` contains all 500 paired outcomes, initial
observation/action digests, source/dependency fingerprints and the calibration
selection. The frozen choice is percentile **99.0**, without BF16 exceptions;
its task-7 calibration holdout speedup was **1.549x**. Task 5 remains weak on
both arms; a separate upstream closed-loop comparison is needed before
attributing that weakness to the checkpoint or engine.

`sm120_checkpoint_reference_results.json` independently checks **812 loaded
checkpoint tensors** and compares three real inputs against official LeRobot
with identical explicit noise values. The minimum action cosine is **0.999927**.
This reference check is numerical, not another 500-episode task campaign.

### Reproduction protocol

The calibration study takes eight fixed demonstration frames per task and eight
disjoint numeric holdout frames. It compares percentile 99.0/99.9/100 and BF16
exceptions for encoder down-projection scale outliers. Selection happens before
closed-loop evaluation. These percentiles reduce *per-frame maxima across
frames*, not individual activation elements. An outlier warning alone is not
proof of activation saturation or task failure.

FP8 weights are uploaded and quantized one matrix at a time from the CPU
checkpoint. The whole BF16 model is no longer staged on the GPU. Explicit
`bf16_encoder_down_layers=(...)` exceptions are available on
`Pi05TorchFrontendRtx` for single-sample inference; batched mode refuses this
uncertified combination. `flash_rt.load_model(..., action_horizon=10)` explicitly
declares the public pi0.5 horizon, and other horizons fail before model loading.
For recalibration, call `model.calibrate(..., prompt=...)` with the new fixed
sample set. The legacy `model.recalibrate()` cache-reset shortcut is not supported
by this checkpoint frontend.

Use a source checkout and a dedicated Python 3.10 environment. The complete
tested dependency list is `requirements-sm120.lock`; it includes a TorchCodec
version compatible with Torch 2.9.0+cu128. OpenPI's required transformers changes
are installed from commit `215abfb217dbac7d5f1273282331b9b1866c0479`, with all five
files SHA-256 checked (the model's version check is not bypassed).

```bash
uv venv --python 3.10 --seed /path/to/pi05-env
uv pip install --python /path/to/pi05-env/bin/python --torch-backend cu128 \
  -r examples/pi05_vla/requirements-sm120.lock ./serving
/path/to/pi05-env/bin/python scripts/install_pi05_transformers.py
source /path/to/pi05-env/bin/activate
```

Build the SM120 `flash_rt_kernels` and `flash_rt_fa2` targets with the same Python
interpreter if this checkout does not already contain matching native binaries.
The manual GPU workflow shows the full CMake invocation. Download the
fixed calibration shards (roughly 500 MB), then prepare an isolated simulator
configuration. Older `hf` versions spell `--type dataset` as `--repo-type dataset`.

```bash
hf download lerobot/libero_spatial_image --type dataset \
  --revision d86c0b94922572b3b657e1d1a3d01f0952ddeb46 \
  --include 'data/chunk-000/file-00[0-4].parquet' meta/tasks.parquet \
  --local-dir /path/to/libero_spatial

PYTHONPATH=serving:. python scripts/prepare_pi05_libero.py \
  --install --download --assets /path/to/libero-assets \
  --config-dir /path/to/isolated-libero-config

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=serving:. OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LIBERO_CONFIG_PATH=/path/to/isolated-libero-config python \
  examples/pi05_vla/reproduce_sm120_libero.py \
  --checkpoint /path/to/pi05_libero --dataset /path/to/libero_spatial \
  --output /path/to/new-campaign
```

The wrapper invokes the same campaign as the driver's `--matched-config`
entry. Results, source/binary/asset hashes, calibration rows, per-episode noise
and action digests, and logs stay in the output directory. Repeating the command
resumes completed task arms; changed configuration or sources require a new
directory. All 500 matched pairs are required for a qualification result. The
non-inferiority margin is 5 percentage points. The existing repository Tango
matched-pair score implementation supplies a one-sided 95% lower decision bound
and a separately computed central 95% interval. A task where Native succeeds at
least once and FP8 never succeeds also fails. The scope is this fixed ten-task suite.
Pass `--evidence examples/pi05_vla/sm120_libero_matched_results.json` to the wrapper
to reproduce the published calibration choice without a new candidate selection;
the source, environment and regenerated calibration manifest must match.

CPU CI checks the evidence and matching rules. The manual
`.github/workflows/pi05-sm120.yml` workflow builds the current kernels and runs
real numeric/startup-memory gates on a trusted runner labelled `rtx5090`; the
full LIBERO campaign is an explicit workflow input. Provision the FlashRT CUDA
environment and set `PI05_CHECKPOINT`, `PI05_DATASET`, and `LIBERO_CONFIG_PATH`
repository variables on that runner. Runner registration is a separate GitHub
administrative step. No external runner is created by adding the workflow file.

The pi0.5 adapter wheel includes its numeric/reference/matched/historical-screen/diagnostic
JSON evidence, dependency lock and reproduction scripts under
`share/instinctflash/pi05`. The installed-package
check verifies those assets are present.

### Task-5 three-arm diagnosis

The completed 50-seed comparison measured **official LeRobot 29/50 (58%)**,
**FlashRT BF16 28/50 (56%)**, and **FlashRT FP8 34/50 (68%)**. FP8 minus
official is +10 percentage points, with a paired central 95% interval of
[-4.54, +24.75] points. This does not establish superiority. The weakness is
also present in the official model under this shared protocol; it cannot be
attributed solely to FP8 quantization.

The failure traces contain both ineffective/wrong-object grasps and placement
errors. Of the 21 official failures, 13 lifted the target and reached within
7 cm of the plate, one lifted but stayed farther away, and seven never lifted
the target 4 cm. All 21 ended outside the actual 3 cm horizontal success
threshold. For example, seed 40122 moved the ramekin and the other bowl while
the target bowl was not lifted; the videos and body-position traces distinguish
this from a failure to place an otherwise correctly grasped target.

**Reproducibility finding:** all 50 BF16 action traces, success outcomes and
noise sequences matched the earlier campaign. FP8 did not: fresh calibration
changed 39 actual scale values despite identical frames and percentile, all
50 action digests differed, and seven seeds changed success outcome
(four gains / three losses; 33/50 previously, 34/50 now). Common noise prefixes
still matched; different episode lengths account for full noise-list differences.
Autotune logs also differed, but this diagnosis does not isolate their causal
contribution. Persisting/validating actual calibration scales and selected GEMM
algorithms is a remaining requirement for bit-exact cross-process FP8 replay.
The earlier 500-pair certificate remains an immutable record of that run, not
a promise that recalibrating/re-autotuning reproduces identical trajectories.

The separate `benchmarks.vla.pi05_sm120_diagnostic` tool compares official
LeRobot, FlashRT BF16 and FlashRT FP8 without changing the qualified rollout or
inference implementation. It reuses a completed 500-pair campaign's checkpoint,
calibration frames, selected percentile, frozen resets and seeds 40100–40149.
Each arm runs in its own process: computed chunk 50, executed horizon 10,
NFE 10, 10 settling steps and a 280-step limit. Official weights are checked
tensor-by-tensor, including the legitimate tied embedding alias.

The official arm receives the exact BF16-rounded diffusion noise values used
by FlashRT, promoted to FP32 for its stock action projection. This isolates
implementation differences but is **not** an evaluation of LeRobot's default
FP32 random sampler. All arms share the qualified simulator/input adapter, so
a shared adapter issue cannot be excluded by this comparison alone.

Use the locked environment and EGL variables from the reproduction instructions
above, then run from the repository root:

```bash
PYTHONPATH=serving:. python -m benchmarks.vla.pi05_sm120_diagnostic \
  --baseline /path/to/completed-500-pair-campaign \
  --output /path/to/new-task5-diagnostic --task 5 --episodes 50

PYTHONPATH=. python scripts/analyze_pi05_diagnostic.py \
  --campaign /path/to/new-task5-diagnostic \
  --output /path/to/diagnostic-analysis --sheet-seeds 40102 40122 \
  --evidence /path/to/task5-diagnostic-evidence.json
```

Each episode saves an MP4 with both cameras and a non-pickle NPZ containing
policy observations, BF16-rounded noise, predicted chunks, executed actions,
robot states, body positions and the actual success predicates. The report
checks their hashes and recomputes the action/noise digests. Complete arms can
be resumed; partial arms are rerun. Changed sources/configuration require a new
output directory. The diagnostic refuses to write into the baseline campaign.

`sm120_task5_diagnostic_results.json` is the compact portable record; videos and
bulk trajectories remain outside the repository. CPU CI checks its source
hashes, all 50 paired seeds, noise sequences and recomputed confidence intervals.
It is a **post-hoc task diagnosis**, not a new suite-wide qualification or a
claim of superiority. Recording overhead makes its timings unsuitable for a
performance certificate. Geometric labels (4 cm lift / 7 cm proximity) are
descriptive only: they never replace LIBERO's real `On` predicate, which also
requires contact, vertical ordering and horizontal center separation below 3 cm.

## Run it

```bash
pip install ./examples/pi05_vla
instinctflash plan  <a-checkpoint-declaring-backbone-pi05>
instinctflash run   <a-checkpoint-declaring-backbone-pi05>
```

`plan` needs no weights and no GPU. `run` needs the patched transformers described above, plus a GPU
with room for 14.5 GB of weights.

On RTX 5090, the public Runtime reproduction also passed with `lerobot/pi05_base` revision
`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` (model SHA-256
`0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`): all 812 remapped
weight keys loaded, the static-KV capture installed, and six queued actions were finite and
advanced through the checkpoint-native 50-action chunk. A pinned local snapshot avoids a second
Hub download:

```bash
IFL_PI05_BASE=/path/to/pi05_base python examples/pi05_vla/run_pi05_end_to_end.py
```

## Choose precision on Thor

The same Runtime interface supports native precision and explicit FP8:

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("lerobot/pi05_base", precision="fp8")
```

Omit `precision` for the native default. In the CLI, add `--fp8` to
`instinctflash serve lerobot/pi05_base`. A numeric tier ceiling alone does not
select FP8; an explicit bit-exact ceiling conflicts with FP8.

The measured BASE and LIBERO v044 routes retain native processing, ten denoise
steps, the checkpoint's 50-action queue and reset behavior. Their native dtypes
and workloads differ, so compare speed within each checkpoint. The v044 paired
LIBERO screen measured native 16/20 and FP8 17/20 successes; this small sample
does not establish no loss, and does not certify BASE task quality. Current
speed measurements, startup costs and limitations are in the
[Thor comparison](../../eval/thor_precision_completion_2026-09-09/COMPARISON.md).

## Attribution

`lerobot/pi05_base` and LeRobot are Apache-2.0, © The HuggingFace Inc. team. Nothing is copied here —
this adapter reads that checkpoint's published configuration and declares it in InstinctFlash's terms.
