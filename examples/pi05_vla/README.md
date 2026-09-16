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

The checked-in gate used four real calibration frames and three held-out rows/seeds. All three
FP8 action chunks cleared the preregistered cosine floor of 0.98; the minimum was **0.9999183** and
the maximum absolute action delta was **0.021599**. Median replay latency was
**31.85 ms native → 20.60 ms FP8 (1.546x)**. FP8 registered 253 quantized weights. The
one-time quantization peak remains 9.15 GiB, but the frontend then releases 15 fully replaced BF16
source tensors (5.04 GiB). Steady allocated memory is **6.33 GiB native vs 3.83 GiB FP8**, a
2.50 GiB (39.4%) reduction; the qualification gate caps the FP8/native resident ratio at 0.75.

This is the FlashRT `pi05` operating point: a 10-action, 7-dimensional LIBERO horizon. It is not
the LeRobot Runtime adapter's checkpoint-native 50-action queue, which is separately exercised by
`run_pi05_end_to_end.py`. Reproduce the SM120 gate with:

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

The same FP8 binary and checkpoint completed a real `libero_spatial` simulator screen using
`hf-libero==0.1.4`, robosuite 1.4.0, MuJoCo 3.8.1, and the immutable
`lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` assets. Ten tasks × three
episodes produced **21/30 successes (70.0%)**. Per-task successes were
`[2, 1, 3, 3, 2, 2, 3, 0, 3, 2] / 3`; every task loaded, rendered non-blank EGL observations,
calibrated, captured, and completed without a runtime failure.

This is deliberately labelled a **SCREEN**: three episodes per task are not a statistical
non-inferiority certificate, and this run did not include a matched native-precision arm. The
complete protocol, package/source/binary hashes, task descriptions, timings, and limitations are
in `sm120_libero_screen_results.json`. Reproduce after configuring the pinned LIBERO assets with:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
LIBERO_CONFIG_PATH=/path/to/libero-config PYTHONPATH=serving:. python \
  serving/examples/thor/eval_libero.py \
  --checkpoint /path/to/pi05_libero --task_suite libero_spatial \
  --framework torch --num_trials 3 --replan_steps 5 --seed 7 \
  --output /tmp/pi05-libero-sm120-screen.json
```

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
