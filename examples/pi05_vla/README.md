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

## Run it

```bash
pip install ./examples/pi05_vla
instinctflash plan  <a-checkpoint-declaring-backbone-pi05>
instinctflash run   <a-checkpoint-declaring-backbone-pi05>
```

`plan` needs no weights and no GPU. `run` needs the patched transformers described above, plus a GPU
with room for 14.5 GB of weights.

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
LIBERO screen measured native 16/20 successes and FP8 on the engine in
`instinctflash/engines/pi05` 15/20 on the same scenes and seeds. On twelve
recorded observations with identical inputs and noise, the FP8 engine's mean
absolute action error against the native FP32 reference was 0.0199 in the
checkpoint's normalized action space (native BF16: 0.0046). These small samples
do not establish no loss, and do not certify BASE task quality.

## LIBERO latency benchmarks

Two standalone examples compare stock pi0.5 eager inference with InstinctFlash's
loop-constant hoists and static-KV CUDA graphs using real LIBERO observations.
Both default to `lerobot/pi05_libero_finetuned_v044` and use one GPU.

Use an environment with working LeRobot pi0.5 and LIBERO installations, the
patched transformers required above, CUDA-enabled PyTorch, and InstinctFlash
with this example package installed. LIBERO assets and checkpoint/tokenizer
access must also be available. Run the commands below from the repository root
with that environment activated. The wrappers require `CUDA_VISIBLE_DEVICES`;
choose one idle GPU. Set `IFL_PI05_PY=/path/to/python` to select another interpreter.

### Fixed real observation

```bash
CUDA_VISIBLE_DEVICES=0 examples/pi05_vla/reproduce_if_fixed_obs_libero.sh
```

This resets LIBERO object task 0 once and prepares its camera, task, and state
inputs through LeRobot's processors. The simulator is then closed. Both arms
reuse that observation and the same seeded noise for repeated chunk predictions.
Each arm reports the median of 15 timed chunks after two unmeasured chunks;
InstinctFlash also completes graph-capture warmup before its measurements.

Each timed chunk includes prefix embedding/prefill, seeded noise creation and
transfer, and ten denoising steps. Environment creation and observation
preprocessing are excluded. Results and a same-input action-equality check are
printed to the terminal; this script does not save a report or evaluate task
success. Set `IFL_PI05_CKPT` to override its checkpoint.

### Live episodes

```bash
CUDA_VISIBLE_DEVICES=0 examples/pi05_vla/reproduce_sample_actions.sh \
  --output-dir ./outputs/pi05-live-libero
```

The output directory must not already exist. Omit `--output-dir` for terminal-only
results. Additional options are available with:

```bash
CUDA_VISIBLE_DEVICES=0 examples/pi05_vla/reproduce_sample_actions.sh --help
```

Defaults are LIBERO object task 0, batch size one, five measured episodes per arm,
and episode seeds starting at 1000. Stock runs first, then InstinctFlash, with the
same episode seeds and initial-state sequence. Each episode advances the real
simulator using postprocessed predicted actions until termination, truncation,
or the environment's step limit. The checkpoint determines the action chunk,
execution horizon, and denoising-step count; for the default checkpoint these
are 50, 50, and 10 respectively.

The timer surrounds `model.sample_actions(...)`, with CUDA synchronization
before and after. It includes noise generation, prefix embedding/prefill, and
denoising integration. Observation preparation, action postprocessing, queued
action execution, simulator stepping, and logging are excluded. Stock uses two
unmeasured chunk predictions and InstinctFlash uses three, without simulator
steps; these are warmup **chunks**. Capture must pass its
exactness check and replay before IF measurements are accepted.

Both arms disable `compile_model`. The live script enables
`torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.benchmark` for
both arms. These settings are not explicitly configured by the fixed-observation
script, which follows the original `reproduce_h100.py` setup.

The terminal shows warmup progress, episode summaries, and final metrics. With
`--output-dir`, `results.json` also contains arguments, per-call latencies,
episode outcomes, capture evidence, and aggregate statistics. It is updated after
each completed episode; failures during evaluation are recorded when possible.
The reported median pools all measured fresh chunk predictions, rather than
averaging episode medians. Speedup is stock median divided by IF median.
Matching episode starts and success counts do not establish per-call bit-exact
outputs across separate live rollouts.

For example, one RTX A6000 run of the default live setup measured 163.6 → 114.7
ms (1.43×), with 18 predictions per arm. This is an
illustrative measurement, not a performance or task-quality guarantee.

CPU-only helper tests require pytest and can be run without model weights or a
LIBERO simulator:

```bash
python -m pytest tests/test_pi05_libero_latency.py -q
```

## Attribution

`lerobot/pi05_base` and LeRobot are Apache-2.0, © The HuggingFace Inc. team. Nothing is copied here —
this adapter reads that checkpoint's published configuration and declares it in InstinctFlash's terms.
